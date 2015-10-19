from __future__ import absolute_import

import numpy as np
import ufl

from pyop2 import op2
from pyop2.profiling import timed_region
from pyop2.utils import as_tuple

import firedrake.dmplex as dmplex
import firedrake.fiat_utils as fiat_utils
import firedrake.utils as utils
from firedrake.petsc import PETSc


__all__ = ['Mesh']


class _Facets(object):
    """Wrapper class for facet interation information on a :class:`Mesh`

    .. warning::

       The unique_markers argument **must** be the same on all processes."""
    def __init__(self, mesh, classes, kind, facet_cell, local_facet_number, markers=None,
                 unique_markers=None):

        self.mesh = mesh

        classes = as_tuple(classes, int, 4)
        self.classes = classes

        self.kind = kind
        assert(kind in ["interior", "exterior"])
        if kind == "interior":
            self._rank = 2
        else:
            self._rank = 1

        self.facet_cell = facet_cell

        self.local_facet_number = local_facet_number

        # assert that markers is a proper subset of unique_markers
        if markers is not None:
            for marker in markers:
                assert (marker in unique_markers), \
                    "Every marker has to be contained in unique_markers"

        self.markers = markers
        self.unique_markers = [] if unique_markers is None else unique_markers
        self._subsets = {}

    @utils.cached_property
    def set(self):
        size = self.classes
        halo = None
        # TODO:
        # if isinstance(self.mesh, ExtrudedMesh):
        #     if self.kind == "interior":
        #         base = self.mesh._old_mesh.interior_facets.set
        #     else:
        #         base = self.mesh._old_mesh.exterior_facets.set
        #     return op2.ExtrudedSet(base, layers=self.mesh.layers)
        return op2.Set(size, "%s_%s_facets" % (self.mesh.name, self.kind), halo=halo)

    @property
    def bottom_set(self):
        '''Returns the bottom row of cells.'''
        return self.mesh.cell_set

    @utils.cached_property
    def _null_subset(self):
        '''Empty subset for the case in which there are no facets with
        a given marker value. This is required because not all
        markers need be represented on all processors.'''

        return op2.Subset(self.set, [])

    def measure_set(self, integral_type, subdomain_id):
        '''Return the iteration set appropriate to measure. This will
        either be for all the interior or exterior (as appropriate)
        facets, or for a particular numbered subdomain.'''

        # ufl.Measure doesn't have enums for these any more :(
        if subdomain_id in ["everywhere", "otherwise"]:
            if integral_type == "exterior_facet_bottom":
                return [(op2.ON_BOTTOM, self.bottom_set)]
            elif integral_type == "exterior_facet_top":
                return [(op2.ON_TOP, self.bottom_set)]
            elif integral_type == "interior_facet_horiz":
                return self.bottom_set
            else:
                return self.set
        else:
            return self.subset(subdomain_id)

    def subset(self, markers):
        """Return the subset corresponding to a given marker value.

        :param markers: integer marker id or an iterable of marker ids"""
        if self.markers is None:
            return self._null_subset
        markers = as_tuple(markers, int)
        try:
            return self._subsets[markers]
        except KeyError:
            # check that the given markers are valid
            for marker in markers:
                if marker not in self.unique_markers:
                    raise LookupError(
                        '{0} is not a valid marker'.
                        format(marker))

            # build a list of indices corresponding to the subsets selected by
            # markers
            indices = np.concatenate([np.nonzero(self.markers == i)[0]
                                      for i in markers])
            self._subsets[markers] = op2.Subset(self.set, indices)
            return self._subsets[markers]

    @utils.cached_property
    def local_facet_dat(self):
        """Dat indicating which local facet of each adjacent
        cell corresponds to the current facet."""

        return op2.Dat(op2.DataSet(self.set, self._rank), self.local_facet_number,
                       np.uintc, "%s_%s_local_facet_number" % (self.mesh.name, self.kind))

    @utils.cached_property
    def facet_cell_map(self):
        """Map from facets to cells."""
        return op2.Map(self.set, self.bottom_set, self._rank, self.facet_cell,
                       "facet_to_cell_map")


class Mesh(object):
    """A representation of mesh topology."""

    def __init__(self, plex, name, reorder, distribute):
        utils._init()

        self._plex = plex
        self.name = name

        dim = plex.getDimension()

        cStart, cEnd = plex.getHeightStratum(0)  # cells
        cell_nfacets = plex.getConeSize(cStart)

        self._ufl_cell = ufl.Cell(fiat_utils._cells[dim][cell_nfacets])

        def callback(self):
            del self._callback
            if op2.MPI.comm.size > 1:
                self._plex.distributeOverlap(1)

            if reorder:
                with timed_region("Mesh: reorder"):
                    old_to_new = self._plex.getOrdering(PETSc.Mat.OrderingType.RCM).indices
                    reordering = np.empty_like(old_to_new)
                    reordering[old_to_new] = np.arange(old_to_new.size, dtype=old_to_new.dtype)
            else:
                # No reordering
                reordering = None

            # Mark OP2 entities and derive the resulting Plex renumbering
            with timed_region("Mesh: renumbering"):
                dmplex.mark_entity_classes(self._plex)
                self._entity_classes = dmplex.get_entity_classes(self._plex)
                self._plex_renumbering = dmplex.plex_renumbering(self._plex,
                                                                 self._entity_classes,
                                                                 reordering)

            with timed_region("Mesh: cell numbering"):
                # Derive a cell numbering from the Plex renumbering
                entity_dofs = np.zeros(dim+1, dtype=np.int32)
                entity_dofs[-1] = 1

                self._cell_numbering = self._plex.createSection([1], entity_dofs,
                                                                perm=self._plex_renumbering)
                entity_dofs[:] = 0
                entity_dofs[0] = 1
                self._vertex_numbering = self._plex.createSection([1], entity_dofs,
                                                                  perm=self._plex_renumbering)

        self._callback = callback

    def init(self):
        """Finish the initialisation of the mesh."""
        if hasattr(self, '_callback'):
            self._callback(self)

    def ufl_cell(self):
        """The UFL :class:`~ufl.cell.Cell` associated with the mesh."""
        return self._ufl_cell

    @utils.cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        plex = self._plex
        dim = plex.getDimension()

        # Cell numbering and global vertex numbering
        cell_numbering = self._cell_numbering
        vertex_numbering = self._vertex_numbering.createGlobalSection(plex.getPointSF())

        cellname = self.ufl_cell().cellname()
        if cellname in ufl.cell.affine_cells:
            # Simplex mesh
            cStart, cEnd = plex.getHeightStratum(0)
            a_closure = plex.getTransitiveClosure(cStart)[0]

            entity_per_cell = np.zeros(dim + 1, dtype=np.int32)
            for dim in xrange(dim + 1):
                start, end = plex.getDepthStratum(dim)
                entity_per_cell[dim] = sum(map(lambda idx: start <= idx < end,
                                               a_closure))

            return dmplex.closure_ordering(plex, vertex_numbering,
                                           cell_numbering, entity_per_cell)

        elif cellname == "quadrilateral":
            # Quadrilateral mesh
            cell_ranks = dmplex.get_cell_remote_ranks(plex)

            facet_orientations = dmplex.quadrilateral_facet_orientations(
                plex, vertex_numbering, cell_ranks)

            cell_orientations = dmplex.orientations_facet2cell(
                plex, vertex_numbering, cell_ranks,
                facet_orientations, cell_numbering)

            dmplex.exchange_cell_orientations(plex,
                                              cell_numbering,
                                              cell_orientations)

            return dmplex.quadrilateral_closure_ordering(
                plex, vertex_numbering, cell_numbering, cell_orientations)

        else:
            raise NotImplementedError("Cell type '%s' not supported." % cellname)

    @utils.cached_property
    def exterior_facets(self):
        if self._plex.getStratumSize("exterior_facets", 1) > 0:
            # Compute the facet_numbering

            # Order exterior facets by OP2 entity class
            exterior_facets, exterior_facet_classes = \
                dmplex.get_facets_by_class(self._plex, "exterior_facets")

            # Derive attached boundary IDs
            if self._plex.hasLabel("boundary_ids"):
                boundary_ids = np.zeros(exterior_facets.size, dtype=np.int32)
                for i, facet in enumerate(exterior_facets):
                    boundary_ids[i] = self._plex.getLabelValue("boundary_ids", facet)

                unique_ids = np.sort(self._plex.getLabelIdIS("boundary_ids").indices)
            else:
                boundary_ids = None
                unique_ids = None

            exterior_local_facet_number, exterior_facet_cell = \
                dmplex.facet_numbering(self._plex, "exterior",
                                       exterior_facets,
                                       self._cell_numbering,
                                       self.cell_closure)

            return _Facets(self, exterior_facet_classes, "exterior",
                           exterior_facet_cell, exterior_local_facet_number,
                           boundary_ids, unique_markers=unique_ids)
        else:
            if self._plex.hasLabel("boundary_ids"):
                unique_ids = np.sort(self._plex.getLabelIdIS("boundary_ids").indices)
            else:
                unique_ids = None
            return _Facets(self, 0, "exterior", None, None,
                           unique_markers=unique_ids)

    @utils.cached_property
    def interior_facets(self):
        if self._plex.getStratumSize("interior_facets", 1) > 0:
            # Compute the facet_numbering

            # Order interior facets by OP2 entity class
            interior_facets, interior_facet_classes = \
                dmplex.get_facets_by_class(self._plex, "interior_facets")

            interior_local_facet_number, interior_facet_cell = \
                dmplex.facet_numbering(self._plex, "interior",
                                       interior_facets,
                                       self._cell_numbering,
                                       self.cell_closure)

            return _Facets(self, interior_facet_classes, "interior",
                           interior_facet_cell, interior_local_facet_number)
        else:
            return _Facets(self, 0, "interior", None, None)

    def make_cell_node_list(self, global_numbering, fiat_element):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg fiat_element: The FIAT element for the cell
        """
        return dmplex.get_cell_nodes(global_numbering,
                                     self.cell_closure,
                                     fiat_element)

    # @property
    # def layers(self):
    #     """Return the number of layers of the extruded mesh
    #     represented by the number of occurences of the base mesh."""
    #     return self._layers

    # TODO: cell_orientations

    # def num_cells(self):
    #     cStart, cEnd = self._plex.getHeightStratum(0)
    #     return cEnd - cStart

    # def num_facets(self):
    #     fStart, fEnd = self._plex.getHeightStratum(1)
    #     return fEnd - fStart

    # def num_faces(self):
    #     fStart, fEnd = self._plex.getDepthStratum(2)
    #     return fEnd - fStart

    # def num_edges(self):
    #     eStart, eEnd = self._plex.getDepthStratum(1)
    #     return eEnd - eStart

    # def num_vertices(self):
    #     vStart, vEnd = self._plex.getDepthStratum(0)
    #     return vEnd - vStart

    # def num_entities(self, d):
    #     eStart, eEnd = self._plex.getDepthStratum(d)
    #     return eEnd - eStart

    # def size(self, d):
    #     return self.num_entities(d)

    def cell_dimension(self):
        """Return the cell dimension"""
        return self.ufl_cell().topological_dimension()

    def facet_dimension(self):
        """Returns the facet dimension."""
        # Facets have co-dimension 1
        return self.ufl_cell().topological_dimension() - 1

    @property
    def cell_classes(self):
        return self._entity_classes[self.cell_dimension(), :]

    @utils.cached_property
    def cell_set(self):
        size = list(self.cell_classes)
        return op2.Set(size, "%s_cells" % self.name)

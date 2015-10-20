from __future__ import absolute_import

import numpy as np
import ufl
import weakref

import coffee.base as ast

from pyop2 import op2
from pyop2.profiling import timed_region
from pyop2.utils import as_tuple

import firedrake.extrusion_utils as eutils
import firedrake.dmplex as dmplex
import firedrake.fiat_utils as fiat_utils
import firedrake.halo as halo
import firedrake.utils as utils
import firedrake.vector as vector
from firedrake.petsc import PETSc


__all__ = ['Mesh', 'ExtrudedMesh']


valuetype = np.float64


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
        if isinstance(self.mesh, ExtrudedMesh):
            if self.kind == "interior":
                base = self.mesh._base_mesh.interior_facets.set
            else:
                base = self.mesh._base_mesh.exterior_facets.set
            return op2.ExtrudedSet(base, layers=self.mesh.layers)
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

        # Mark exterior and interior facets
        # Note.  This must come before distribution, because otherwise
        # DMPlex will consider facets on the domain boundary to be
        # exterior, which is wrong.
        with timed_region("Mesh: label facets"):
            label_boundary = (op2.MPI.comm.size == 1) or distribute
            dmplex.label_facets(plex, label_boundary=label_boundary)

        # Distribute the dm to all ranks
        if op2.MPI.comm.size > 1 and distribute:
            # We distribute with overlap zero, in case we're going to
            # refine this mesh in parallel.  Later, when we actually use
            # it, we grow the halo.
            plex.distribute(overlap=0)

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

    @property
    def layers(self):
        return None

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

    def make_global_numbering(self, dofs_per_entity):
        # Create the PetscSection mapping topological entities to DoFs
        return self._plex.createSection([1], dofs_per_entity,
                                        perm=self._plex_renumbering)

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

    def num_vertices(self):
        vStart, vEnd = self._plex.getDepthStratum(0)
        return vEnd - vStart

    # def num_entities(self, d):
    #     eStart, eEnd = self._plex.getDepthStratum(d)
    #     return eEnd - eStart

    # def size(self, d):
    #     return self.num_entities(d)

    def cell_dimension(self):
        """Returns the cell dimension."""
        return self.ufl_cell().topological_dimension()

    def facet_dimension(self):
        """Returns the facet dimension."""
        # Facets have co-dimension 1
        return self.ufl_cell().topological_dimension() - 1

    @utils.cached_property
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "%s_cells" % self.name)


class ExtrudedMesh(Mesh):
    """Build an extruded mesh from an input mesh

    :arg mesh:           the unstructured base mesh
    :arg layers:         number of extruded cell layers in the "vertical"
                         direction.
    """

    def __init__(self, mesh, layers):
        mesh.init()

        self._base_mesh = mesh
        if layers < 1:
            raise RuntimeError("Must have at least one layer of extruded cells (not %d)" % layers)
        # All internal logic works with layers of base mesh (not layers of cells)
        self._layers = layers + 1
        self._ufl_cell = ufl.OuterProductCell(mesh.ufl_cell(), ufl.Cell("interval", 1))

        self._plex = mesh._plex
        self._plex_renumbering = mesh._plex_renumbering
        # TODO:
        # self._entity_classes = mesh._entity_classes
        # self.name = mesh.name
        # self._cell_numbering = mesh._cell_numbering

    @property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        return self._base_mesh.cell_closure

    @utils.cached_property
    def exterior_facets(self):
        exterior_facets = self._base_mesh.exterior_facets
        return _Facets(self, exterior_facets.classes,
                       "exterior",
                       exterior_facets.facet_cell,
                       exterior_facets.local_facet_number,
                       exterior_facets.markers,
                       unique_markers=exterior_facets.unique_markers)

    @utils.cached_property
    def interior_facets(self):
        interior_facets = self._base_mesh.interior_facets
        return _Facets(self, interior_facets.classes,
                       "interior",
                       interior_facets.facet_cell,
                       interior_facets.local_facet_number)

    def make_cell_node_list(self, global_numbering, fiat_element):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg fiat_element: The FIAT element for the cell
        """
        return dmplex.get_cell_nodes(global_numbering,
                                     self.cell_closure,
                                     fiat_utils.FlattenedElement(fiat_element))

    @property
    def layers(self):
        """Return the number of layers of the extruded mesh
        represented by the number of occurences of the base mesh."""
        return self._layers

    def cell_dimension(self):
        """Returns the cell dimension."""
        return (self._base_mesh.cell_dimension(), 1)

    def facet_dimension(self):
        """Returns the facet dimension.

        .. note::

            This only returns the dimension of the "side" (vertical) facets,
            not the "top" or "bottom" (horizontal) facets.

        """
        return (self._base_mesh.facet_dimension(), 1)

    @utils.cached_property
    def cell_set(self):
        return op2.ExtrudedSet(self._base_mesh.cell_set, layers=self.layers)


class FunctionSpaceBase(object):
    """Base class for function spaces."""

    def __init__(self, mesh, element, name=None, shape=()):
        """
        :param mesh: :class:`Mesh` to build this space on
        :param element: :class:`ufl.FiniteElementBase` to build this space from
        :param name: user-defined name for this space
        :param shape: shape of a :class:`.VectorFunctionSpace` or :class:`.TensorFunctionSpace`
        """

        assert mesh.ufl_cell() == element.cell()

        self._mesh = mesh
        self._ufl_element = element
        self.name = name
        self.shape = shape

        # Compute the FIAT version of the UFL element above
        self.fiat_element = fiat_utils.fiat_from_ufl_element(element)

        if isinstance(mesh, ExtrudedMesh):
            # Set up some extrusion-specific things

            # Get the flattened version of the FIAT element
            flattened_element = fiat_utils.FlattenedElement(self.fiat_element)

            # Compute the dofs per column
            dofs_per_entity = eutils.compute_extruded_dofs(self.fiat_element,
                                                           flattened_element.entity_dofs(),
                                                           mesh.layers)

            # Compute the offset for the extrusion process
            self.offset = eutils.compute_offset(self.fiat_element.entity_dofs(),
                                                self.flattened_element.entity_dofs(),
                                                self.fiat_element.space_dimension())

            # Compute the top and bottom masks to identify boundary dofs
            #
            # Sorting the keys of the closure entity dofs, the whole cell
            # comes last [-1], before that the horizontal facet [-2], before
            # that vertical facets [-3]. We need the horizontal facets here.
            closure_dofs = self.fiat_element.entity_closure_dofs()
            b_mask = closure_dofs[sorted(closure_dofs.keys())[-2]][0]
            t_mask = closure_dofs[sorted(closure_dofs.keys())[-2]][1]
            self.bt_masks = {}
            self.bt_masks["topological"] = (b_mask, t_mask)  # conversion to tuple
            # Geometric facet dofs
            facet_dofs = self.fiat_element.horiz_facet_support_dofs()
            self.bt_masks["geometric"] = (facet_dofs[0], facet_dofs[1])

            self.extruded = True
        else:
            # If not extruded specific, set things to None/False, etc.
            self.offset = None
            self.bt_masks = None
            self.extruded = False

            entity_dofs = self.fiat_element.entity_dofs()
            dofs_per_entity = [len(entity[0]) for d, entity in entity_dofs.iteritems()]

        dm = PETSc.DMShell().create()
        dm.setAttr('__fs__', weakref.ref(self))
        dm.setPointSF(mesh._plex.getPointSF())
        # Create the PetscSection mapping topological entities to DoFs
        sec = mesh.make_global_numbering(dofs_per_entity)
        dm.setDefaultSection(sec)
        self._global_numbering = sec
        self._dm = dm
        self._ises = None
        self._halo = halo.Halo(dm)

        # Compute entity class offsets
        self.dof_classes = [0, 0, 0, 0]
        for d in range(mesh._plex.getDimension()+1):
            ndofs = dofs_per_entity[d]
            for i in range(4):
                self.dof_classes[i] += ndofs * mesh._entity_classes[d, i]

        # Tell the DM about the layout of the global vector
        # TODO:
        # from firedrake.function import Function
        # with Function(self).dat.vec_ro as v:
        #     self._dm.setGlobalVector(v.duplicate())

        self._node_count = self._global_numbering.getStorageSize()

        self.cell_node_list = mesh.make_cell_node_list(self._global_numbering,
                                                       self.fiat_element)

        if mesh._plex.getStratumSize("interior_facets", 1) > 0:
            self.interior_facet_node_list = \
                dmplex.get_facet_nodes(mesh.interior_facets.facet_cell,
                                       self.cell_node_list)
        else:
            self.interior_facet_node_list = np.array([], dtype=np.int32)

        if mesh._plex.getStratumSize("exterior_facets", 1) > 0:
            self.exterior_facet_node_list = \
                dmplex.get_facet_nodes(mesh.exterior_facets.facet_cell,
                                       self.cell_node_list)
        else:
            self.exterior_facet_node_list = np.array([], dtype=np.int32)

        # Empty map caches. This is a sui generis cache
        # implementation because of the need to support boundary
        # conditions.
        self._cell_node_map_cache = {}
        self._exterior_facet_map_cache = {}
        self._interior_facet_map_cache = {}

    # @property
    # def index(self):
    #     """Position of this :class:`FunctionSpaceBase` in the
    #     :class:`.MixedFunctionSpace` it was extracted from."""
    #     return self._index

    @property
    def node_count(self):
        """The number of global nodes in the function space. For a
        plain :class:`.FunctionSpace` this is equal to
        :attr:`dof_count`, however for a :class:`.VectorFunctionSpace`,
        the :attr:`dof_count`, is :attr:`dim` times the
        :attr:`node_count`."""

        return self._node_count

    @property
    def dof_count(self):
        """The number of global degrees of freedom in the function
        space. Cf. :attr:`node_count`."""

        return self.node_count*self.dim

    @utils.cached_property
    def node_set(self):
        """A :class:`pyop2.Set` containing the nodes of this
        :class:`.FunctionSpace`. One or (for
        :class:`.VectorFunctionSpace`\s) more degrees of freedom are
        stored at each node.
        """

        name = "%s_nodes" % self.name
        if self._halo:
            s = op2.Set(self.dof_classes, name,
                        halo=self._halo)
            if self.extruded:
                return op2.ExtrudedSet(s, layers=self._mesh.layers)
            return s
        else:
            s = op2.Set(self.node_count, name)
            if self.extruded:
                return op2.ExtrudedSet(s, layers=self._mesh.layers)
            return s

    @utils.cached_property
    def dof_dset(self):
        """A :class:`pyop2.DataSet` containing the degrees of freedom of
        this :class:`.FunctionSpace`."""
        return op2.DataSet(self.node_set, self.dim, name="%s_nodes_dset" % self.name)

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.Dat` defined on the
        :attr:`dof_dset` of this :class:`.Function`."""
        return op2.Dat(self.dof_dset, val, valuetype, name, uid=uid)

    def cell_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.cell_node_map()
        else:
            parent = None

        return self._map_cache(self._cell_node_map_cache,
                               self._mesh.cell_set,
                               self.cell_node_list,
                               self.fiat_element.space_dimension(),
                               bcs,
                               "cell_node",
                               self.offset,
                               parent)

    def interior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.interior_facet_node_map()
        else:
            parent = None

        offset = self.cell_node_map().offset
        map = self._map_cache(self._interior_facet_map_cache,
                              self._mesh.interior_facets.set,
                              self.interior_facet_node_list,
                              2*self.fiat_element.space_dimension(),
                              bcs,
                              "interior_facet_node",
                              offset=np.append(offset, offset),
                              parent=parent)
        map.factors = (self._mesh.interior_facets.facet_cell_map,
                       self.cell_node_map())
        return map

    def exterior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from exterior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.exterior_facet_node_map()
        else:
            parent = None

        facet_set = self._mesh.exterior_facets.set
        if isinstance(self._mesh, ExtrudedMesh):
            name = "extruded_exterior_facet_node"
            offset = self.offset
        else:
            name = "exterior_facet_node"
            offset = None
        return self._map_cache(self._exterior_facet_map_cache,
                               facet_set,
                               self.exterior_facet_node_list,
                               self.fiat_element.space_dimension(),
                               bcs,
                               name,
                               parent=parent,
                               offset=offset)

    def bottom_nodes(self, method='topological'):
        """Return a list of the bottom boundary nodes of the extruded mesh.
        The bottom mask is applied to every bottom layer cell to get the
        dof ids."""
        try:
            mask = self.bt_masks[method][0]
        except KeyError:
            raise ValueError("Unknown boundary condition method %s" % method)
        return np.unique(self.cell_node_list[:, mask])

    def top_nodes(self, method='topological'):
        """Return a list of the top boundary nodes of the extruded mesh.
        The top mask is applied to every top layer cell to get the dof ids."""
        try:
            mask = self.bt_masks[method][1]
        except KeyError:
            raise ValueError("Unknown boundary condition method %s" % method)
        voffs = self.offset.take(mask)*(self._mesh.layers-2)
        return np.unique(self.cell_node_list[:, mask] + voffs)

    def _map_cache(self, cache, entity_set, entity_node_list, map_arity, bcs, name,
                   offset=None, parent=None):
        if bcs is not None:
            # Separate explicit bcs (we just place negative entries in
            # the appropriate map values) from implicit ones (extruded
            # top and bottom) that require PyOP2 code gen.
            explicit_bcs = [bc for bc in bcs if bc.sub_domain not in ['top', 'bottom']]
            implicit_bcs = [(bc.sub_domain, bc.method) for bc in bcs if bc.sub_domain in ['top', 'bottom']]
            if len(explicit_bcs) == 0:
                # Implicit bcs are not part of the cache key for the
                # map (they only change the generated PyOP2 code),
                # hence rewrite bcs here.
                bcs = None
            if len(implicit_bcs) == 0:
                implicit_bcs = None
        else:
            implicit_bcs = None
        if bcs is None:
            # Empty tuple if no bcs found.  This is so that matrix
            # assembly, which uses a set to keep track of the bcs
            # applied to matrix hits the cache when that set is
            # empty.  tuple(set([])) == tuple().
            lbcs = tuple()
        else:
            for bc in bcs:
                fs = bc.function_space()
                # TODO:
                # if isinstance(fs, IndexedVFS):
                #     fs = fs._parent
                if fs != self:
                    raise RuntimeError("DirichletBC defined on a different FunctionSpace!")
            # Ensure bcs is a tuple in a canonical order for the hash key.
            lbcs = tuple(sorted(bcs, key=lambda bc: bc.__hash__()))
        try:
            # Cache hit
            val = cache[lbcs]
            # In the implicit bc case, we decorate the cached map with
            # the list of implicit boundary conditions so PyOP2 knows
            # what to do.
            if implicit_bcs:
                val = op2.DecoratedMap(val, implicit_bcs=implicit_bcs)
            return val
        except KeyError:
            # Cache miss.

            # Any top and bottom bcs (for the extruded case) are handled elsewhere.
            nodes = [bc.nodes for bc in lbcs if bc.sub_domain not in ['top', 'bottom']]
            decorate = False
            if nodes:
                bcids = reduce(np.union1d, nodes)
                negids = np.copy(bcids)
                for bc in lbcs:
                    if bc.sub_domain in ["top", "bottom"]:
                        continue
                    # TODO:
                    # if isinstance(bc.function_space(), IndexedVFS):
                    #     # For indexed VFS bcs, we encode the component
                    #     # in the high bits of the map value.
                    #     # That value is then negated to indicate to
                    #     # the generated code to discard the values
                    #     #
                    #     # So here we do:
                    #     #
                    #     # node = -(node + 2**(30-cmpt) + 1)
                    #     #
                    #     # And in the generated code we can then
                    #     # extract the information to discard the
                    #     # correct entries.
                    #     val = 2 ** (30 - bc.function_space().index)
                    #     # bcids is sorted, so use searchsorted to find indices
                    #     idx = np.searchsorted(bcids, bc.nodes)
                    #     negids[idx] += val
                    #     decorate = True
                node_list_bc = np.arange(self.node_count, dtype=np.int32)
                # Fix up for extruded, doesn't commute with indexedvfs for now
                if isinstance(self.mesh(), ExtrudedMesh):
                    node_list_bc[bcids] = -10000000
                else:
                    node_list_bc[bcids] = -(negids + 1)
                new_entity_node_list = node_list_bc.take(entity_node_list)
            else:
                new_entity_node_list = entity_node_list

            val = op2.Map(entity_set, self.node_set,
                          map_arity,
                          new_entity_node_list,
                          ("%s_"+name) % (self.name),
                          offset,
                          parent,
                          self.bt_masks)

            if decorate:
                val = op2.DecoratedMap(val, vector_index=True)
            cache[lbcs] = val
            if implicit_bcs:
                return op2.DecoratedMap(val, implicit_bcs=implicit_bcs)
            return val

    @utils.memoize
    def exterior_facet_boundary_node_map(self, method):
        '''The :class:`pyop2.Map` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.

        :arg method: The method for determining boundary nodes. See
            :class:`~.bcs.DirichletBC`.
        '''

        el = self.fiat_element

        dim = self._mesh.facet_dimension()

        if method == "topological":
            boundary_dofs = el.entity_closure_dofs()[dim]
        elif method == "geometric":
            if self.extruded:
                # This function is only called on extruded meshes when
                # asking for the nodes that live on the "vertical"
                # exterior facets.  Hence we don't need to worry about
                # horiz_facet_support_dofs as well.
                boundary_dofs = el.vert_facet_support_dofs()
            else:
                boundary_dofs = el.facet_support_dofs()

        nodes_per_facet = \
            len(boundary_dofs[0])

        # HACK ALERT
        # The facet set does not have a halo associated with it, since
        # we only construct halos for DoF sets.  Fortunately, this
        # loop is direct and we already have all the correct
        # information available locally.  So We fake a set of the
        # correct size and carry out a direct loop
        facet_set = op2.Set(self._mesh.exterior_facets.set.total_size)

        fs_dat = op2.Dat(facet_set**el.space_dimension(),
                         data=self.exterior_facet_node_map().values_with_halo)

        facet_dat = op2.Dat(facet_set**nodes_per_facet,
                            dtype=np.int32)

        local_facet_nodes = np.array(
            [dofs for e, dofs in boundary_dofs.iteritems()])

        # Helper function to turn the inner index of an array into c
        # array literals.
        c_array = lambda xs: "{"+", ".join(map(str, xs))+"}"

        body = ast.Block([ast.Decl("int",
                                   ast.Symbol("l_nodes", (len(el.get_reference_element().topology[dim]),
                                                          nodes_per_facet)),
                                   init=ast.ArrayInit(c_array(map(c_array, local_facet_nodes))),
                                   qualifiers=["const"]),
                          ast.For(ast.Decl("int", "n", 0),
                                  ast.Less("n", nodes_per_facet),
                                  ast.Incr("n", 1),
                                  ast.Assign(ast.Symbol("facet_nodes", ("n",)),
                                             ast.Symbol("cell_nodes", ("l_nodes[facet[0]][n]",))))
                          ])

        kernel = op2.Kernel(ast.FunDecl("void", "create_bc_node_map",
                                        [ast.Decl("int*", "cell_nodes"),
                                         ast.Decl("int*", "facet_nodes"),
                                         ast.Decl("unsigned int*", "facet")],
                                        body),
                            "create_bc_node_map")

        local_facet_dat = op2.Dat(facet_set ** self._mesh.exterior_facets._rank,
                                  self._mesh.exterior_facets.local_facet_dat.data_ro_with_halos,
                                  dtype=np.uintc)
        op2.par_loop(kernel, facet_set,
                     fs_dat(op2.READ),
                     facet_dat(op2.WRITE),
                     local_facet_dat(op2.READ))

        if isinstance(self._mesh, ExtrudedMesh):
            offset = self.offset[boundary_dofs[0]]
        else:
            offset = None
        return op2.Map(facet_set, self.node_set,
                       nodes_per_facet,
                       facet_dat.data_ro_with_halos,
                       name="exterior_facet_boundary_node",
                       offset=offset)

    @property
    def dim(self):
        """The product of the :attr:`.dim` of the :class:`.FunctionSpace`."""
        return np.prod(self.shape, dtype=int)

    def mesh(self):
        return self._mesh

    def ufl_element(self):
        return self._ufl_element

    # def __len__(self):
    #     return 1

    # def __iter__(self):
    #     yield self

    # def __getitem__(self, i):
    #     """Return ``self`` if ``i`` is 0 or raise an exception."""
    #     if i != 0:
    #         raise IndexError("Only index 0 supported on a FunctionSpace")
    #     return self

    # def __mul__(self, other):
    #     """Create a :class:`.MixedFunctionSpace` composed of this
    #     :class:`.FunctionSpace` and other"""
    #     return MixedFunctionSpace((self, other))


class FunctionSpace(FunctionSpaceBase):
    """TODO"""

    def __init__(self, mesh, element, name=None):
        mesh.init()
        super(FunctionSpace, self).__init__(mesh, element, name=name)

    # def __getitem__(self, i):
    #     """Return self if ``i`` is 0, otherwise raise an error."""
    #     assert i == 0, "Can only extract subspace 0 from %r" % self
    #     return self


class VectorFunctionSpace(FunctionSpaceBase):
    """TODO"""

    def __init__(self, mesh, element, dim, name=None):
        mesh.init()
        super(VectorFunctionSpace, self).__init__(mesh, element, name=name, shape=(dim,))

    # def __getitem__(self, i):
    #     """Return self if ``i`` is 0, otherwise raise an error."""
    #     assert i == 0, "Can only extract subspace 0 from %r" % self
    #     return self

    # def sub(self, i):
    #     """Return an :class:`IndexedVFS` for the requested component.

    #     This can be used to apply :class:`~.DirichletBC`\s to components
    #     of a :class:`VectorFunctionSpace`."""
    #     return IndexedVFS(self, i)


class Function(ufl.Coefficient):
    """TODO"""

    def __init__(self, function_space, val=None, name=None, dtype=valuetype):
        """
        :param function_space: the :class:`.FunctionSpace`, :class:`.VectorFunctionSpace`
            or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
            Alternatively, another :class:`Function` may be passed here and its function space
            will be used to build this :class:`Function`.
        :param val: NumPy array-like (or :class:`op2.Dat`) providing initial values (optional).
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to :data:`valuetype`).
        """

        assert isinstance(function_space, FunctionSpaceBase)
        self._function_space = function_space

        ufl.Coefficient.__init__(self, self._function_space.ufl_element())

        self.uid = utils._new_uid()
        self._name = name or 'function_%d' % self.uid

        if isinstance(val, (op2.Dat, op2.DatView)):
            self.dat = val
        else:
            self.dat = self._function_space.make_dat(val, dtype,
                                                     self._name, uid=self.uid)

        # self._split = None

    # def split(self):
    #     """Extract any sub :class:`Function`\s defined on the component spaces
    #     of this this :class:`Function`'s :class:`FunctionSpace`."""
    #     if self._split is None:
    #         self._split = tuple(Function(fs, dat, name="%s[%d]" % (self.name(), i))
    #                             for i, (fs, dat) in
    #                             enumerate(zip(self._function_space, self.dat)))
    #     return self._split

    # def sub(self, i):
    #     """Extract the ith sub :class:`Function` of this :class:`Function`.

    #     :arg i: the index to extract

    #     See also :meth:`split`.

    #     If the :class:`Function` is defined on a
    #     :class:`.~VectorFunctionSpace`, this returns a proxy object
    #     indexing the ith component of the space, suitable for use in
    #     boundary condition application."""
    #     if isinstance(self.function_space(), functionspace.VectorFunctionSpace):
    #         fs = self.function_space().sub(i)
    #         return Function(fs, val=op2.DatView(self.dat, i),
    #                         name="view[%d](%s)" % (i, self.name()))
    #     return self.split()[i]

    # @property
    # def cell_set(self):
    #     """The :class:`pyop2.Set` of cells for the mesh on which this
    #     :class:`Function` is defined."""
    #     return self._function_space._mesh.cell_set

    # @property
    # def node_set(self):
    #     """A :class:`pyop2.Set` containing the nodes of this
    #     :class:`Function`. One or (for
    #     :class:`.VectorFunctionSpace`\s) more degrees of freedom are
    #     stored at each node.
    #     """
    #     return self._function_space.node_set

    # @property
    # def dof_dset(self):
    #     """A :class:`pyop2.DataSet` containing the degrees of freedom of
    #     this :class:`Function`."""
    #     return self._function_space.dof_dset

    # def cell_node_map(self, bcs=None):
    #     return self._function_space.cell_node_map(bcs)
    # cell_node_map.__doc__ = functionspace.FunctionSpace.cell_node_map.__doc__

    # def interior_facet_node_map(self, bcs=None):
    #     return self._function_space.interior_facet_node_map(bcs)
    # interior_facet_node_map.__doc__ = functionspace.FunctionSpace.interior_facet_node_map.__doc__

    # def exterior_facet_node_map(self, bcs=None):
    #     return self._function_space.exterior_facet_node_map(bcs)
    # exterior_facet_node_map.__doc__ = functionspace.FunctionSpace.exterior_facet_node_map.__doc__

    def vector(self):
        """Return a :class:`.Vector` wrapping the data in this :class:`Function`"""
        return vector.Vector(self.dat)

    def function_space(self):
        """Return the :class:`.FunctionSpace`, :class:`.VectorFunctionSpace`
            or :class:`.MixedFunctionSpace` on which this :class:`Function` is defined."""
        return self._function_space

    def name(self):
        """Return the name of this :class:`Function`"""
        return self._name

    # def rename(self, name=None, label=None):
    #     """Set the name and or label of this :class:`Function`

    #     :arg name: The new name of the `Function` (if not `None`)
    #     :arg label: The new label for the `Function` (if not `None`)
    #     """
    #     if name is not None:
    #         self._name = name
    #     if label is not None:
    #         self._label = label

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return super(Function, self).__str__()

    # def assign(self, expr, subset=None):
    #     """Set the :class:`Function` value to the pointwise value of
    #     expr. expr may only contain :class:`Function`\s on the same
    #     :class:`.FunctionSpace` as the :class:`Function` being assigned to.

    #     Similar functionality is available for the augmented assignment
    #     operators `+=`, `-=`, `*=` and `/=`. For example, if `f` and `g` are
    #     both Functions on the same :class:`FunctionSpace` then::

    #       f += 2 * g

    #     will add twice `g` to `f`.

    #     If present, subset must be an :class:`pyop2.Subset` of
    #     :attr:`node_set`. The expression will then only be assigned
    #     to the nodes on that subset.
    #     """

    #     if isinstance(expr, Function) and \
    #             expr._function_space == self._function_space:
    #         expr.dat.copy(self.dat, subset=subset)
    #         return self

    #     from firedrake import assemble_expressions
    #     assemble_expressions.evaluate_expression(
    #         assemble_expressions.Assign(self, expr), subset)

    #     return self

    # def __iadd__(self, expr):

    #     if np.isscalar(expr):
    #         self.dat += expr
    #         return self
    #     if isinstance(expr, Function) and \
    #             expr._function_space == self._function_space:
    #         self.dat += expr.dat
    #         return self

    #     from firedrake import assemble_expressions
    #     assemble_expressions.evaluate_expression(
    #         assemble_expressions.IAdd(self, expr))

    #     return self

    # def __isub__(self, expr):

    #     if np.isscalar(expr):
    #         self.dat -= expr
    #         return self
    #     if isinstance(expr, Function) and \
    #             expr._function_space == self._function_space:
    #         self.dat -= expr.dat
    #         return self

    #     from firedrake import assemble_expressions
    #     assemble_expressions.evaluate_expression(
    #         assemble_expressions.ISub(self, expr))

    #     return self

    # def __imul__(self, expr):

    #     if np.isscalar(expr):
    #         self.dat *= expr
    #         return self
    #     if isinstance(expr, Function) and \
    #             expr._function_space == self._function_space:
    #         self.dat *= expr.dat
    #         return self

    #     from firedrake import assemble_expressions
    #     assemble_expressions.evaluate_expression(
    #         assemble_expressions.IMul(self, expr))

    #     return self

    # def __idiv__(self, expr):

    #     if np.isscalar(expr):
    #         self.dat /= expr
    #         return self
    #     if isinstance(expr, Function) and \
    #             expr._function_space == self._function_space:
    #         self.dat /= expr.dat
    #         return self

    #     from firedrake import assemble_expressions
    #     assemble_expressions.evaluate_expression(
    #         assemble_expressions.IDiv(self, expr))

    #     return self

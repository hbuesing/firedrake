from __future__ import absolute_import

from firedrake.petsc import PETSc
from firedrake import mesh
from firedrake import op2
import h5py


__all__ = ["CheckpointFile", "FileMode"]


"""Available modes for opening checkpoint files"""
FileMode = PETSc.Viewer.Mode


class CheckpointFile(object):

    """Open a file that can be used for checkpointing.

    Data will be written in HDF5 format.

    :arg name: The name of the file to be used"""
    def __init__(self, name, comm=None):
        self.name = name
        self.vwr = None
        self.comm = comm or op2.MPI.comm
        self._stored_mesh = False
        self._dm = None

    def open_vwr(self, mode):
        """Open a PETSc Viewer for this file with specified mode.

        :arg mode: The mode to open the file in.  The possible modes
             can been found as attributes of the :data:`~.FileMode`
             object.
        """
        self.close()
        self.vwr = PETSc.ViewerHDF5().create(self.name, mode=mode)

    def write_metadata(self):
        f = h5py.File(self.name, mode="w")
        f.require_group('/metadata')
        f['metadata']['numprocs'] = self.comm.size
        f.close()

    def read_metadata(self):
        f = h5py.File(self.name, mode="r")
        assert self.comm.size == f['metadata']['numprocs'].value, \
            "Redistribution not yet implemented"
        f.close()

    def store_mesh(self, mesh):
        if self._stored_mesh:
            return
        self.write_metadata()
        self.open_vwr(FileMode.APPEND)
        dm = mesh._plex
        # Kill these, we don't want them in the output
        for label in ["op2_core",
                      "op2_non_core",
                      "op2_exec_halo",
                      "op2_non_exec_halo"]:
            dm.setLabelOutput(label, False)
        dm.view(self.vwr)
        self._dm = dm
        self._stored_mesh = True

    def store_function(self, function):
        assert self._stored_mesh, "Must call store_mesh first"
        self.vwr.pushGroup("/fields")
        with function.dat.vec_ro as v:
            nat = v.duplicate()
            nat.setName(function.name())
            self._dm.globalToNaturalBegin(v, nat)
            self._dm.globalToNaturalEnd(v, nat)
            nat.view(self.vwr)
        self.vwr.popGroup()

    def close(self):
        if self.vwr is not None:
            self.vwr.destroy()
            self.vwr = None

    def load_mesh(self):
        assert self.vwr is None
        self.read_metadata()
        self.open_vwr(FileMode.READ)
        dm = PETSc.DMPlex().create()
        dm.load(self.vwr)
        return mesh.Mesh(dm, reorder=True)

    def load_function(self, f):
        assert self.vwr is not None

        self.vwr.pushGroup("/fields")
        with f.dat.vec as v:
            v.setName(f.name())
            v.load(self.vwr)

        self.vwr.popGroup()

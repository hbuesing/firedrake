from __future__ import absolute_import

from firedrake.petsc import PETSc
from firedrake import mesh
from firedrake import op2
import h5py


__all__ = ["Checkpoint", "DumbCheckpoint", "READ", "CREATE", "UPDATE"]


class _Mode(object):
    """Mode used for opening files"""

    def __init__(self, h5py_mode, petsc_mode):
        self.hmode = h5py_mode
        self.pmode = petsc_mode

FileMode = PETSc.Viewer.Mode

"""Open a checkpoint file for reading.  Errors if file does not exist."""
READ = _Mode("r", FileMode.READ)

"""Create a checkpoint file.  Truncates the file if it exists."""
CREATE = _Mode("w", FileMode.WRITE)

"""Open a checkpoint file for updating.  Creates the file if it does
not exist, otherwise gives READ and WRITE access."""
UPDATE = _Mode("a", FileMode.APPEND)


class DumbCheckpoint(object):

    """A very dumb checkpoint object.

    This checkpoint object is capable of writing :class:`~.Function`\s
    to disk in parallel (using HDF5) and reloading them on the same
    number of processes and a :class:`~.Mesh` constructed identically.

    :arg name: the name of the checkpoint file.
    :arg mode: the access mode (one of :data:`~.READ`,
         :data:`~.CREATE`, or :data:`~.UPDATE`)
    :arg comm: (optional) communicator the writes should be collective
         over.

    This object can be used in a context manager (in which case it
    closes the file when the scope is exited).

    """
    def __init__(self, name, mode=UPDATE, comm=None):
        self.comm = comm or op2.MPI.comm
        # Read (or write) metadata on rank 0.
        if self.comm.rank == 0:
            with h5py.File(name, mode=mode.hmode) as f:
                group = f.require_group("/metadata")
                dset = group.get("numprocs", None)
                if dset is None:
                    group["numprocs"] = self.comm.size
                elif dset.value != self.comm.size:
                    nprocs = self.comm.bcast(dset.value, root=0)
        else:
            nprocs = self.comm.bcast(None, root=0)

        # Verify
        if nprocs != self.comm.size:
            raise ValueError("Process mismatch: written on %d, have %d" %
                             (nprocs, self.comm.size))
        # Now switch to UPDATE if we were asked to CREATE the file
        # (we've already created it above, and CREATE truncates)
        if mode is CREATE:
            mode = UPDATE
        self.vwr = PETSc.ViewerHDF5().create(name, mode=mode.pmode)

    def close(self):
        """Close the checkpoint file (flushing any pending writes)"""
        if hasattr(self, "vwr"):
            self.vwr.destroy()

    def store(self, function, name=None):
        with function.dat.vec_ro as v:
            self.vwr.pushGroup("/fields")
            v.setName(name or function.name())
            v.view(self.vwr)
            self.vwr.popGroup()

    def load(self, function, name=None):
        with function.dat.vec as v:
            self.vwr.pushGroup("/fields")
            v.setName(name or function.name())
            v.load(self.vwr)
            self.vwr.popGroup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Checkpoint(object):

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
            v.setName(function.name())
            v.view(self.vwr)
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
        return mesh.Mesh(dm, reorder=False)

    def load_function(self, f):
        assert self.vwr is not None

        self.vwr.pushGroup("/fields")
        with f.dat.vec as v:
            v.setName(f.name())
            v.load(self.vwr)

        self.vwr.popGroup()

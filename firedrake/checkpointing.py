from __future__ import absolute_import
from firedrake.petsc import PETSc
from firedrake import op2
from firedrake import hdf5interface as h5i
import firedrake


__all__ = ["DumbCheckpoint", "FILE_READ", "FILE_CREATE", "FILE_UPDATE"]


"""Open a checkpoint file for reading.  Raises an error if file does not exist."""
FILE_READ = PETSc.Viewer.Mode.READ

"""Create a checkpoint file.  Truncates the file if it exists."""
FILE_CREATE = PETSc.Viewer.Mode.WRITE

"""Open a checkpoint file for updating.  Creates the file if it does
not exist, providing both read and write access."""
FILE_UPDATE = PETSc.Viewer.Mode.APPEND


class DumbCheckpoint(object):

    """A very dumb checkpoint object.

    This checkpoint object is capable of writing :class:`~.Function`\s
    to disk in parallel (using HDF5) and reloading them on the same
    number of processes and a :class:`~.Mesh` constructed identically.

    :arg name: the name of the checkpoint file.
    :arg mode: the access mode (one of :data:`~.FILE_READ`,
         :data:`~.FILE_CREATE`, or :data:`~.FILE_UPDATE`)
    :arg comm: (optional) communicator the writes should be collective
         over.

    This object can be used in a context manager (in which case it
    closes the file when the scope is exited).

    """
    def __init__(self, name, mode=FILE_UPDATE, comm=None):
        self.comm = comm or op2.MPI.comm
        if mode == FILE_READ:
            import os
            if not os.path.exists(name):
                raise IOError("File '%s' does not exist, cannot be opened for reading" % name)
        self.vwr = PETSc.ViewerHDF5().create(name, mode=mode, comm=self.comm)

    def close(self):
        """Close the checkpoint file (flushing any pending writes)"""
        if hasattr(self, "vwr"):
            self.vwr.destroy()

    def store(self, function, name=None):
        """Store a function in the checkpoint file.

        :arg function: The function to store.
        :arg name: an (optional) name to store the function under.  If
             not provided, uses :data:`function.name()`.
        """
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only store functions")
        name = name or function.name()
        with function.dat.vec_ro as v:
            self.vwr.pushGroup("/fields")
            oname = v.getName()
            v.setName(name)
            v.view(self.vwr)
            v.setName(oname)
            self.vwr.popGroup()
            # Write metadata
            self.write_attribute("/fields/%s" % name, "nprocs", self.comm.size)

    def write_attribute(self, obj, name, val):
        """Set an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.
        :arg val: The attribute value.

        Raises :exc:`AttributeError` if writing the attribute fails.

        .. note::

           Only ``int``-valued attributes are supported.
        """
        if self.has_attribute(obj, name):
            raise AttributeError("Cannot overwrite existing attribute '%s' on '%s'" %
                                 (name, obj))
        h5i.write_attribute(self.vwr, obj, name, val)

    def read_attribute(self, obj, name):
        """Read an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.

        Raises :exec:`AttributeError` if reading the attribute fails.

        .. note::

           Only ``int``-valued attributes are supported.
        """
        if not self.has_attribute(obj, name):
            raise AttributeError("Attribute '%s' on '%s' not found" % (name, obj))
        return h5i.read_attribute(self.vwr, obj, name)

    def has_attribute(self, obj, name):
        """Check for existance of an HDF5 attribute on a specified data object.

        :arg obj: The path to the data object.
        :arg name: The name of the attribute.
        """
        return h5i.has_attribute(self.vwr, obj, name)

    def load(self, function, name=None):
        """Store a function from the checkpoint file.

        :arg function: The function to load values into.
        :arg name: an (optional) name used to find the function values.  If
             not provided, uses :data:`function.name()`.
        """
        if not isinstance(function, firedrake.Function):
            raise ValueError("Can only load functions")
        name = name or function.name()
        nprocs = self.read_attribute("/fields/%s" % name, "nprocs")
        if nprocs is not None and nprocs != self.comm.size:
            raise ValueError("Process mismatch: written on %d, have %d" %
                             (nprocs, self.comm.size))
        with function.dat.vec as v:
            self.vwr.pushGroup("/fields")
            oname = v.getName()
            v.setName(name)
            v.load(self.vwr)
            v.setName(oname)
            self.vwr.popGroup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

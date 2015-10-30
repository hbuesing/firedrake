from __future__ import absolute_import
from firedrake.petsc import PETSc
from firedrake import op2
from firedrake import hdf5interface as h5i
import firedrake


__all__ = ["DumbCheckpoint", "FILE_READ", "FILE_CREATE", "FILE_UPDATE"]


FILE_READ = PETSc.Viewer.Mode.READ
"""Open a checkpoint file for reading.  Raises an error if file does not exist."""

FILE_CREATE = PETSc.Viewer.Mode.WRITE
"""Create a checkpoint file.  Truncates the file if it exists."""

FILE_UPDATE = PETSc.Viewer.Mode.APPEND
"""Open a checkpoint file for updating.  Creates the file if it does not exist, providing both read and write access."""


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
        self.mode = mode
        self.vwr = PETSc.ViewerHDF5().create(name, mode=mode, comm=self.comm)

    def close(self):
        """Close the checkpoint file (flushing any pending writes)"""
        if hasattr(self, "_h5rep"):
            del self._h5rep
        if hasattr(self, "vwr"):
            self.vwr.destroy()
            del self.vwr

    def __del__(self):
        self.close()

    def store(self, function, name=None):
        """Store a function in the checkpoint file.

        :arg function: The function to store.
        :arg name: an (optional) name to store the function under.  If
             not provided, uses ``function.name()``.
        """
        if self.mode is FILE_READ:
            raise IOError("Cannot store to checkpoint opened with mode 'FILE_READ'")
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
            obj = "/fields/%s" % name
            name = "nprocs"
            if not self.has_attribute(obj, name):
                self.write_attribute(obj, name, self.comm.size)

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
             not provided, uses ``function.name()``.
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

    def as_h5py(self):
        """Attempt to convert the file handle to a :class:`h5py:File`.

        This fails if h5py was not linked to the same HDF5 library as PETSc.

        .. warning::

           Explicitly closing this file, using :meth:`h5py:File.close`
           will result in the checkpoint file being closed and PETSc
           will probably subsequently produce an error.

        """
        self._h5rep = h5i.get_h5py_file(self.vwr)
        return self._h5rep

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

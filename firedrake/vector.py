from __future__ import absolute_import
import numpy as np
from mpi4py import MPI

from pyop2 import op2

from firedrake.petsc import PETSc


__all__ = ['Vector']


class Vector(object):
    def __init__(self, x):
        """Build a `Vector` that wraps a :class:`pyop2.Dat` for Dolfin compatibilty.

        :arg x: an :class:`pyop2.Dat` to wrap or a :class:`Vector` to copy.
                This copies the underlying data in the :class:`pyop2.Dat`.
        """
        if isinstance(x, Vector):
            self.dat = op2.Dat(x.dat)
        elif isinstance(x, op2.base.Dat):  # ugh
            self.dat = x
        else:
            raise RuntimeError("Don't know how to build a Vector from a %r" % type(x))

    def axpy(self, a, x):
        """Add a*x to self.

        :arg a: a scalar
        :arg x: a :class:`Vector` or :class:`.Function`"""
        self.dat += a*x.dat

    def _scale(self, a):
        """Scale self by `a`.

        :arg a: a scalar (or something that contains a dat)
        """
        try:
            self.dat *= a.dat
        except AttributeError:
            self.dat *= a

    def __mul__(self, other):
        """Scale self by other"""
        return self._scale(other)

    def __add__(self, other):
        """Add other to self"""
        try:
            self.dat += other.dat
        except AttributeError:
            self.dat += other

    def array(self):
        """Return a copy of the process local data as a numpy array"""
        with self.dat.vec_ro as v:
            return np.copy(v.array)

    def get_local(self):
        """Return a copy of the process local data as a numpy array"""
        return self.array()

    def set_local(self, values):
        """Set process local values

        :arg values: a numpy array of values of length :func:`Vector.local_size`"""
        with self.dat.vec as v:
            v.array[:] = values

    def local_size(self):
        """Return the size of the process local data (without ghost points)"""
        return self.dat.dataset.size

    def size(self):
        """Return the global size of the data"""
        if hasattr(self, '_size'):
            return self._size
        lsize = self.local_size()
        self._size = op2.MPI.comm.allreduce(lsize, op=MPI.SUM)
        return self._size

    def inner(self, other):
        """Return the l2-inner product of self with other"""
        return self.dat.inner(other.dat)

    def gather(self, global_indices=None):
        """Gather a :class:`Vector` to all processes

        :arg global_indices: the globally numbered indices to gather
                            (should be the same on all processes).  If
                            `None`, gather the entire :class:`Vector`."""
        if global_indices is None:
            N = self.size()
            v = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
            is_ = PETSc.IS().createStride(N, 0, 1, comm=PETSc.COMM_SELF)
        else:
            global_indices = np.asarray(global_indices, dtype=np.int32)
            N = len(global_indices)
            v = PETSc.Vec().createSeq(N, comm=PETSc.COMM_SELF)
            is_ = PETSc.IS().createGeneral(global_indices, comm=PETSc.COMM_SELF)

        with self.dat.vec_ro as vec:
            vscat = PETSc.Scatter().create(vec, is_, v, None)
            vscat.scatterBegin(vec, v, addv=PETSc.InsertMode.INSERT_VALUES)
            vscat.scatterEnd(vec, v, addv=PETSc.InsertMode.INSERT_VALUES)
        return v.array

    def __setitem__(self, idx, value):
        """Set a value or values in the local data

        :arg idx: the local idx, or indices to set.
        :arg value: the value, or values to give them."""
        self.dat.data[idx] = value

    def __getitem__(self, idx):
        """Return a value or values in the local data

        :arg idx: the local idx, or indices to set."""
        return self.dat.data_ro[idx]

    def __len__(self):
        """Return the length of the local data (not including ghost points)"""
        return self.local_size()

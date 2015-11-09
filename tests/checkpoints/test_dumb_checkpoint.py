import pytest
from firedrake import *
import numpy as np


@pytest.fixture(scope="module",
                params=[False, True],
                ids=["simplex", "quad"])
def mesh(request):
    return UnitSquareMesh(2, 2, quadrilateral=request.param)


@pytest.fixture(params=range(1, 4))
def degree(request):
    return request.param


@pytest.fixture(params=["CG"])
def fs(request):
    return request.param


@pytest.fixture
def dumpfile(tmpdir):
    return str(tmpdir.join("dump.h5"))


@pytest.fixture(scope="module")
def f():
    m = UnitSquareMesh(2, 2)
    V = FunctionSpace(m, 'CG', 1)
    return Function(V, name="f")


def run_store_load(mesh, fs, degree, dumpfile):

    V = FunctionSpace(mesh, fs, degree)

    f = Function(V, name="f")

    expr = Expression("x[0]*x[1]")

    f.interpolate(expr)

    f2 = Function(V, name="f")

    dumpfile = op2.MPI.comm.bcast(dumpfile, root=0)
    chk = DumbCheckpoint(dumpfile, mode=FILE_CREATE)

    chk.store(f)

    chk.load(f2)

    assert np.allclose(f.dat.data_ro, f2.dat.data_ro)


def test_store_load(mesh, fs, degree, dumpfile):
    run_store_load(mesh, fs, degree, dumpfile)


@pytest.mark.parallel(nprocs=2)
def test_store_load_parallel(mesh, fs, degree, dumpfile):
    run_store_load(mesh, fs, degree, dumpfile)


@pytest.mark.parallel(nprocs=2)
def test_serial_checkpoint_parallel_load_fails(f, dumpfile):
    from firedrake.petsc import PETSc
    # Write on COMM_SELF (size == 1)
    chk = DumbCheckpoint("%s.%d" % (dumpfile, op2.MPI.comm.rank),
                         mode=FILE_CREATE, comm=PETSc.COMM_SELF)
    chk.store(f)
    chk.close()
    # Make sure it's written, and broadcast rank-0 name to all processes
    fname = op2.MPI.comm.bcast("%s.0" % dumpfile, root=0)
    with pytest.raises(ValueError):
        with DumbCheckpoint(fname, mode=FILE_READ) as chk:
            # Written on 1 process, loading on 2 should raise ValueError
            chk.load(f)


def test_checkpoint_fails_for_non_function(dumpfile):
    with DumbCheckpoint(dumpfile, mode=FILE_CREATE) as chk:
        with pytest.raises(ValueError):
            chk.store(np.arange(10))


def test_checkpoint_read_not_exist_ioerror(dumpfile):
    with pytest.raises(IOError):
        with DumbCheckpoint(dumpfile, mode=FILE_READ):
            pass


def test_attributes(f, dumpfile):
    mesh = f.function_space().mesh()
    with DumbCheckpoint(dumpfile, mode=FILE_CREATE) as chk:
        with pytest.raises(AttributeError):
            chk.write_attribute("/foo", "nprocs", 1)
        with pytest.raises(AttributeError):
            chk.read_attribute("/bar", "nprocs")

        chk.store(mesh.coordinates, name="coords")

        assert chk.read_attribute("/fields/coords", "nprocs") == 1

        chk.write_attribute("/fields/coords", "dimension",
                            mesh.coordinates.dat.cdim)

        assert chk.read_attribute("/fields/coords", "dimension") == \
            mesh.coordinates.dat.cdim


def test_store_read_only_ioerror(f, dumpfile):
    # Make file
    with DumbCheckpoint(dumpfile, mode=FILE_CREATE) as chk:
        pass
    with DumbCheckpoint(dumpfile, mode=FILE_READ) as chk:
        with pytest.raises(IOError):
            chk.store(f)


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))

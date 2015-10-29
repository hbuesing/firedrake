cimport petsc4py.PETSc as PETSc


cdef extern from "petsc.h":
    ctypedef long PetscInt
    ctypedef enum PetscBool:
        PETSC_TRUE, PETSC_FALSE
    ctypedef enum PetscDataType:
        PETSC_INT
    ctypedef int hid_t


cdef extern from "petscviewerhdf5.h":
    int PetscViewerHDF5WriteAttribute(PETSc.PetscViewer,const char[],const char[],
                                      PetscDataType,const void*)
    int PetscViewerHDF5ReadAttribute(PETSc.PetscViewer,const char[],const char[],
                                     PetscDataType,void*)
    int PetscViewerHDF5HasAttribute(PETSc.PetscViewer,const char[],const char[],
                                    PetscBool*)
    int PetscViewerHDF5GetFileId(PETSc.PetscViewer,hid_t*)


def write_attribute(PETSc.Viewer vwr not None, obj, name, val):
    """Write an HDF5 attribute to a viewer.

    :arg vwr: The PETSc Viewer (must have type HDF5).
    :arg obj: The object to write an attribute on.
    :arg name: The attribute to write.
    :arg val: The attribute value.

    .. note::

       Only supports :data:`int`-valued attributes.
    """
    cdef const char *path = obj
    cdef const char *cname = name
    cdef PetscInt cval = 0
    cdef int ierr = 0
    if vwr.type != vwr.Type.HDF5:
        raise TypeError("Viewer is not an HDF5 viewer")
    if type(val) is not int:
        raise TypeError("Only 'int' attributes supported, not '%s'" % type(val))
    cval = val
    ierr = PetscViewerHDF5WriteAttribute(vwr.vwr, path, cname,
                                         PETSC_INT, <const void *>&cval)
    if ierr != 0:
        raise AttributeError("Unable to write attribute '%s' on object '%s'" % (name, path))


def read_attribute(PETSc.Viewer vwr not None, obj, name):
    """Read an HDF5 attribute from a viewer.

    :arg vwr: The PETSc Viewer (must have type HDF5).
    :arg obj: The object to read an attribute from.
    :arg name: The attribute to return.

    .. note::

       Only supports :data:`int`-valued attributes.
    """
    cdef const char *path = obj
    cdef const char *cname = name
    cdef PetscInt val = 0
    cdef int ierr = 0
    if vwr.type != vwr.Type.HDF5:
        raise TypeError("Viewer is not an HDF5 viewer")
    ierr = PetscViewerHDF5ReadAttribute(vwr.vwr, path, cname, PETSC_INT, <void *>&val)
    if ierr != 0:
        raise AttributeError("Unable to read attribute '%s' on object '%s'" %
                             (name, obj))
    return val


def has_attribute(PETSc.Viewer vwr not None, obj, name):
    """Ascertain if the specified viewer has a particular attribute.

    :arg vwr: The PETSc Viewer (must have type HDF5).
    :arg obj: The object to check an attribute for.
    :arg name: The attribute to check.
    """
    cdef const char *path = obj
    cdef const char *cname = name
    cdef PetscBool flag = PETSC_FALSE
    cdef int ierr = 0
    if vwr.type != vwr.Type.HDF5:
        raise TypeError("Viewer is not an HDF5 viewer")
    ierr = PetscViewerHDF5HasAttribute(vwr.vwr, path, cname, &flag)
    if ierr != 0:
        raise AttributeError("Checking attribute '%s' on object '%s' failed" %
                             (name, obj))
    return <bint>flag


def get_h5py_file(PETSc.Viewer vwr not None):
    """Attempt to convert PETSc viewer file handle to h5py File.

    :arg vwr: The PETSc Viewer (must have type HDF5).

    .. warning::

       For this to work, h5py and PETSc must both have been compiled
       against *the same* HDF5 library (otherwise the file handles are
       not interchangeable).  This is the likeliest reason for failure
       when attempting the conversion."""
    cdef hid_t fid = 0
    cdef int ierr = 0

    if vwr.type != vwr.Type.HDF5:
        raise TypeError("Viewer is not an HDF5 viewer")
    ierr = PetscViewerHDF5GetFileId(vwr.vwr, &fid)
    if ierr != 0:
        raise RuntimeError("Unable to get file handle")

    import h5py
    try:
        objid = h5py.h5i.wrap_identifier(fid)
    except ValueError:
        raise RuntimeError("Unable to convert handle to h5py object. Likely h5py not linked to same HDF5 as PETSc")

    if type(objid) is not h5py.h5f.FileID:
        raise TypeError("Provided handle doesn't reference a file")
    # We got a borrowed reference to the file id from PETSc, need to
    # inc-ref it so that the file isn't closed behind our backs.
    h5py.h5i.inc_ref(objid)
    return h5py.File(objid)

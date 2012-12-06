import numpy as np
cimport numpy as np

np.import_array()


cdef extern from "SLIC.h":
    cdef cppclass SLIC:
        SLIC()
        # img, width, heights, returned labels, num_labels, superpixel size, compactness
        void DoSuperpixelSegmentation_ForGivenSuperpixelSize(unsigned int*, int, int, int *, int, int, double)
        # img, width, heights, returned labels, num_labels, superpixel number, compactness
        void DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(unsigned int*, int, int, int *, int, int, double)
        # img, labeling, width, heights, color for contours
        void DrawContoursAroundSegments(unsigned int** ,int*, int, int, unsigned int)


def slic_s(np.ndarray[np.uint8_t, ndim=3] img, superpixel_size=300, compactness=10):
    """SLIC Superpixels for fixed superpixel size.

    Parameters
    ----------
    img : numpy array, dtype=uint8
        Original image, ARGB (or AXXX) format, A channel is ignored.
        Needs to be C-Contiguous.
    superpixel_size: int, default=300
        Desired size for superpixel
    compactness: douple, default=10
        Degree of compactness of superpixels.

    Returns
    -------
    labels : numpy array

    """

    if (img.shape[2] != 3):
        raise ValueError("Image needs to have 3 channels.")
    if np.isfortran(img):
        raise ValueError("The input image is not C-contiguous")
    cdef np.ndarray[np.uint8_t, ndim=3] img_ = np.empty((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    img_[:, :, 1:] = img
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int * labels
    cdef int n_labels
    cdef SLIC* slic = new SLIC()
    slic.DoSuperpixelSegmentation_ForGivenSuperpixelSize(<unsigned int *>img_.data, w, h,
            labels, n_labels, superpixel_size, compactness)
    cdef np.npy_intp shape[2]
    shape[0] = h
    shape[1] = w
    label_array = np.PyArray_SimpleNewFromData(2, shape, np.NPY_INT32, <void*> labels)
    return label_array


def slic_n(np.ndarray[np.uint8_t, ndim=3] img, n_superpixels=500, compactness=10):
    """SLIC Superpixels for fixed number of superpixels.

    Parameters
    ----------
    img : numpy array, dtype=uint8
        Original image RGB.
        Needs to be C-Contiguous
    n_superpixels: int, default=500
        Desired number of superpixels.
    compactness: douple, default=10
        Degree of compactness of superpixels.

    Returns
    -------
    labels : numpy array

    """
    if (img.shape[2] != 3):
        raise ValueError("Image needs to have 3 channels.")
    if np.isfortran(img):
        raise ValueError("The input image is not C-contiguous")
    cdef np.ndarray[np.uint8_t, ndim=3] img_ = np.empty((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    img_[:, :, :-1] = img
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int * labels
    cdef int n_labels
    cdef SLIC* slic = new SLIC()
    slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(<unsigned int *>img_.data, w, h,
            labels, n_labels, n_superpixels, compactness)
    cdef np.npy_intp shape[2]
    shape[0] = h
    shape[1] = w
    label_array = np.PyArray_SimpleNewFromData(2, shape, np.NPY_INT32, <void*> labels)
    return label_array


def contours(np.ndarray[np.uint8_t, ndim=3] img, np.ndarray[np.int32_t, ndim=2] labels, color=10):
    """Draw contours of superpixels into original image.
    Destoys original!

    Parameters
    ----------
    img : numpy array, dtype=uint8
        Original image.
        Needs to be uint, RGB.
        Needs to be C-Contiguous
    lables: numpy array, dtype=int
        Same width and height as image, 
    color: int
        color for boundaries.
    """
    cdef SLIC* slic = new SLIC()
    assert(img.shape[2] == 3)
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int n_labels
    cdef np.ndarray[np.uint8_t, ndim=3] img_ = np.empty((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    img_[:, :, :-1] = img
    slic.DrawContoursAroundSegments(<unsigned int **>&img_.data, <int*>labels.data, w, h,
            color)
    return img_

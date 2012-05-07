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
        Needs to be C-Contagious
    superpixel_size: int, default=300
        Desired size for superpixel
    compactness: douple, default=10
        Degree of compactness of superpixels.
    
    Returns
    -------
    labels : numpy array
        
    """

    if (img.shape[2] != 4):
        raise ValueError("Image needs to have 4 channels, eventhough the first is ignored.")
    if np.isfortran(img):
        raise ValueError("The input image is not C-contiguous")
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int * labels
    cdef int n_labels
    cdef SLIC* slic = new SLIC()
    slic.DoSuperpixelSegmentation_ForGivenSuperpixelSize(<unsigned int *>img.data, w, h,
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
        Original image ARGB (or AXXX) format, A channel is ignored.
        Needs to be C-Contagious
    n_superpixels: int, default=500
        Desired number of superpixels.
    compactness: douple, default=10
        Degree of compactness of superpixels.
    
    Returns
    -------
    labels : numpy array
        
    """
    if (img.shape[2] != 4):
        raise ValueError("Image needs to have 4 channels, eventhough the first is ignored.")
    if np.isfortran(img):
        raise ValueError("The input image is not C-contiguous")
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int * labels
    cdef int n_labels
    cdef SLIC* slic = new SLIC()
    slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(<unsigned int *>img.data, w, h,
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
        Needs to be uint, ARGB (or AXXX) format, A channel is ignored.
        Needs to be C-Contagious
    lables: numpy array, dtype=int
        Same width and height as image, 
    color: int
        color for boundaries.
    """
    cdef SLIC* slic = new SLIC()
    assert(img.shape[2] == 4)
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int n_labels
    slic.DrawContoursAroundSegments(<unsigned int **>&img.data, <int*>labels.data, w, h,
            color)
    return img

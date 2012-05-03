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


def slic_s(np.ndarray[np.uint8_t, ndim=3] img, superpixel_size, compactness):
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


def slic_n(np.ndarray[np.uint8_t, ndim=3] img, n_superpixels, compactness):
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


def contours(np.ndarray[np.uint8_t, ndim=3] img, np.ndarray[np.int32_t, ndim=2] labels, color):
    cdef SLIC* slic = new SLIC()
    assert(img.shape[2] == 4)
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int n_labels
    slic.DrawContoursAroundSegments(<unsigned int **>&img.data, <int*>labels.data, w, h,
            color)
    return img

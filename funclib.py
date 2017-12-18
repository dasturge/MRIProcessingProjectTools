# library of algorithms used in this skull stripping procedure
# imports
from itertools import product
from itertools import combinations

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


# BEGIN:  Supplementary Functions
def partitions(items, k):

    def split(indices):
        i=0
        for j in indices:
            yield items[i:j]
            i = j
        yield items[i:]

    for indices in combinations(range(1, len(items)), k-1):
        yield list(split(indices))

def slices(indices, lengths):
    return [slice(i, i+j) for i, j in zip(indices, lengths)]


# UNUSED
def DoG(array, sigma1, sigma2=None):
    # difference of gaussians, sigma1 > sigma2
    if not sigma2:
        sigma2 = sigma1 - 1
    return gaussian_filter(array, sigma1) - gaussian_filter(array, sigma2)


def erode(arr, structure):
    """
    returns mask of eroded image
    :param arr: ndarray of data
    :param structure: structuring element on which to erode
    :return: eroded mask of data
    """
    dims = arr.shape
    field = np.array(structure.shape)
    flen = (field - 1) / 2
    flen = flen.astype(int)
    padarr = np.pad(arr, [(x,x) for x in flen], mode='constant', constant_values=0)
    padarr = padarr.astype(np.bool)
    arr = np.zeros_like(arr, dtype=np.bool)
    for indices in np.ndindex(dims):
        ele = padarr[slices(indices, field)]
        arr[indices] = np.all(ele[structure] * structure[structure])

    return arr.astype(np.int8)


def dilate(arr, structure):
    """
    returns mask of dilated image
    :param arr: ndarray of data
    :param structure: structuring element on which to dilate
    :return: dilated mask of data
    """
    dims = arr.shape
    field = np.array(structure.shape)
    flen = (field - 1) / 2
    flen = flen.astype(int)
    padarr = np.pad(arr, [(x,x) for x in flen], mode='constant', constant_values=0)
    padarr = padarr.astype(np.bool)
    arr = np.zeros_like(arr, dtype=np.bool)
    for indices in np.ndindex(dims):
        ele = padarr[slices(indices, field)]
        arr[indices] = np.any(ele[structure] * structure[structure])

    return arr.astype(np.int8)


# BEGIN Functions Utilized in Main
def nii_save_like(array, nii, name):
    """
    saves numpy array input as a nifti with the same orientation metadata as nii
    :param array: ndarray of data
    :param nifti: target nifti in same space loaded in using nibabel
    :param name: name of output file
    :return: None
    """
    img = nib.Nifti1Image(array, nii.affine)
    img.header.from_header(nii.header)
    nib.save(img, name)


def morph_open(arr, structure):
    """
    wraps erode and dilate to perform morphological opening
    :param arr: ndarray of data
    :param structure: structuring element on which to open
    :return: opened data
    """
    return dilate(erode(arr, structure), structure)


def morph_close(arr, structure):
    """
    wraps erode and dilate to perform morphological closing
    :param arr: ndarray of data
    :param structure: structuring element on which to close
    :return: closed data
    """
    return erode(dilate(arr, structure), structure)


def largest_roi(label_arr):
    """
    finds the largest labeled class
    :param label_arr: ndarray of labels (integers) denoting separate classes
    :return: mask of largest class
    """
    counts = np.bincount(label_arr.reshape(label_arr.size))
    maxi = np.argmax(counts[1:]) + 1  # largest excluding 0 (background)
    label_arr = np.where(label_arr == maxi, 1, 0)
    return label_arr


def otsun(arr, n, mask=None):
    """
    performs Otsu's method using n thresholds on arr, returns labeled data.
    :param arr: ndarray of data
    :param n: number of thresholds to use
    :param mask: optional roi on which to perform Otsu's, returns background as class 0.
    :return: ndarray of same shape as data with 0, 1, 2, ..., n - 1 denoting classes.
    If a mask is supplied, returns 0, 1, 2, ..., n with 0 as outside of mask region.
    """
    if mask is None:
        flat = arr.flatten()
    else:
        flat = arr[mask.astype(np.bool)].flatten()
    L = 200
    histogram, edges = np.histogram(flat, bins=200, density=False)
    histogram = histogram / np.sum(histogram)
    moment = histogram * edges[1:]
    mG = np.sum(moment)
    varB = np.zeros((L,)*n)

    # list out possible partitions of histogram, calculate between class variance and store in varB
    for part in partitions(range(L), n+1):
        P = np.fromiter((np.sum(histogram[x]) for x in part), dtype=np.float)
        if 0 in P:
            continue # avoid divide by zero, classes can't have zero elements anyways.
        M = np.fromiter((np.sum(moment[x]) for x in part), dtype=np.float) / P
        idx = tuple(int(x[0] - 1) for x in part[1:])
        varB[idx] = np.sum(P* (M-mG)**2)

    # indices of varB correspond to thresholds by bin number
    idx = np.unravel_index(np.argmax(np.where(np.isnan(varB), 0, varB)), varB.shape)
    otsu = np.zeros_like(arr, np.uint8)
    if mask is not None:
        otsu = otsu + mask.astype(np.bool).astype(np.uint8)
    for i in idx:
        otsu = otsu + np.where(arr > edges[i], 1, 0)

    return otsu


def region_labeling(arr):
    """
    #@todo incomplete
    :param arr:
    :return:
    """
    dims = arr.shape
    labels = np.zeros_like(arr)

    for i, j, k in product(range(dims[0]), range(dims[1]), range(dims[2])):
        if not arr[i, j, k]:
            continue

        elif i > 0 and arr[i - 1, j, k]:
            labels[i, j, k] = labels[i - 1, j, k]

        elif j - 1 < dims[1] and arr[i, j - 1, k]:
            labels[i, j, k] = labels[i, j - 1, k]

        elif k - 1 < dims[2] and arr[i, j, k - 1]:
            labels[i, j, k] = labels[i, j, k - 1]


def cm(arr):
    """
    finds the center of mass of arr
    :param arr: ndarray of data, values are interpreted as weights
    :return: floating point approximation of center of mass
    """
    dims = arr.shape
    moment = np.zeros_like(dims)
    mass = 0

    for indices in np.ndindex(dims):
        if not arr[indices]:
            continue
        mass = mass + arr[indices]
        moment = moment + np.array(indices) * arr[indices]

    center = moment / mass

    return center


def radial_distance(arr, point):
    """
    constructs ndarray of radial distances of nonzero index coordinates to point
    :param arr: ndarray of data
    :param point: reference point
    :return: ndarray shape of arr with distances to point in all nonzero coordinates
    """
    dims = arr.shape
    distance = np.zeros_like(arr)
    for indices in np.ndindex(dims):
        if arr[indices]:
            distance[indices] = np.sqrt(np.sum((np.array(indices) - point)**2))

    return distance

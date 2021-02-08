# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv2

import scipy.ndimage.interpolation as ndii
from scipy import interpolate
from scipy import signal
from scipy import ndimage

import matplotlib.pyplot as plt

import matplotlib.style
import matplotlib as mpl
import math
import time

from heapq import heapify, heappush, heappushpop, nlargest

import PhaseCorrelation as pc

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 450
mpl.rcParams['hatch.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'
"""
def createRaster(params, reference_raster, moving_raster, outfile):
    global args
    #Read reference into single channel image (only necessary to pad to correct width)
    reference = rasterToImage(reference_raster)

    #Read moving_raster into multichannel image
    move_image = moving_raster.ReadAsArray()
    channels = move_image.shape[0]
    move_image = np.dstack( tuple(move_image[i] for i in range(channels)) )

    #Temporary Patch for Bug
    #global new_target_dim
    #move_image = cv2.resize(move_image, new_target_dim)

    r, move_image, bboxr, bboxm = padding(reference, move_image, scaling=2, dtype=np.uint8, bbox=True)
    move_image = post_warp(reference, move_image, params)
    #implot(move_image, "Test of Raster Multicolor")
    #added_plot(reference, move_image, title="Multicolor Test Overlay")

    #r, move_image, bboxr, bboxm = padding(reference, move_image, dtype=np.uint8, bbox=True)

    #Write output raster file
    dst = gdal.GetDriverByName('GTiff').Create(outfile, reference.shape[1], reference.shape[0], channels, gdal.GDT_Byte)
    dst.SetGeoTransform(reference_raster.GetGeoTransform())
    dst.SetProjection(reference_raster.GetProjection())
    for i in range(channels):
        dst.GetRasterBand(i+1).WriteArray(move_image[bboxr[0]:bboxr[1], bboxr[2]:bboxr[3], i])
    dst.FlushCache()
    dst = None
"""

def relative_crop_distance(mbox, cbox):
    corner_distance_x = (cbox[0]-mbox[0])/mbox[2]
    corner_distance_y = (cbox[1]-mbox[1])/mbox[3]
    length_x = (cbox[2]/mbox[2])
    length_y = (cbox[3]/mbox[3])
    return (corner_distance_x, corner_distance_y, length_x, length_y)

def corners_to_box(top, bot, reverse=False):
    if reverse:
        return (top[1], top[0], bot[1]-top[1], bot[0]-top[0])
    else:
        return (top[0], top[1], bot[0]-top[0], bot[1]-top[1])

def corners_to_range(top, bot, reverse=False):
    top = (int(np.floor(top[0])), int(np.floor(top[1])))
    bot = (int(np.ceil(bot[0])), int(np.ceil(bot[1])))
    
    if reverse:
        return (top[1], bot[1], top[0], bot[1])
    else:
        return (top[0], bot[0], top[1], bot[1])

def compute_overlap_box(mask1, mask2):
    joint_mask = np.logical_and(mask1, mask2)
    return compute_bbox(joint_mask)

def compute_crop_box(valid_box, size_threshold, ratio_threshold=None):
    corner = (valid_box[0], valid_box[1])
    size = (valid_box[2], valid_box[3])

    #generate box size
    min_size, max_size = size_threshold
    
    w = np.random.uniform(min_size, max_size)
    
    if ratio_threshold is None:
        h = np.random.uniform(min_size, max_size)
    else:
        top = min(max_size, ratio_threshold*w)
        bot = max(min_size, w / ratio_threshold)
        h = np.random.uniform(bot, top)
        
    #generate box corner
    new_x = np.random.uniform(0, 1-w)
    new_y = np.random.uniform(0, 1-h)
    
    valid_x = size[0] - w
    valid_y = size[1] - h
    
    return relative_locations(valid_box, (new_x, new_y, w, h))

def range_to_crop(loc, image, reverse=False):
    start_x, end_x, start_y, end_y = loc
    if reverse:
        crop = image[start_y:end_y, start_x:end_x]    
    else:
        crop = image[start_x:end_x, start_y:end_y]
    return crop

def range_to_box(rl, reverse=False):
    if reverse:
        return (rl[2], rl[0], rl[3]-rl[2], rl[1]-rl[0])
    else:
        return (rl[0], rl[2], rl[1]-rl[0], rl[3]-rl[2])

def box_to_crop(bbox, image, reverse=False):
    corner = (bbox[0], bbox[1])
    size = (bbox[2], bbox[3])
    if reverse:
        crop = image[corner[1]:corner[1]+size[1], corner[0]:corner[0]+size[0]]
    else:
        crop = image[corner[0]:corner[0]+size[0], corner[1]:corner[1]+size[1]]
    return crop
def range_to_masking_crop(loc, image, reverse=False):
    mask = np.zeros(image.shape)
    start_x, end_x, start_y, end_y = loc
    if reverse:
        mask[start_y:end_y, start_x:end_x] = image[start_y:end_y, start_x:end_x]   
    else:
        mask[start_x:end_x, start_y:end_y] = image[start_x:end_x, start_y:end_y]
    return mask


def loc_to_box(loc, reverse=False):
    rl, ml = loc
    if reverse:
        rbbox = (rl[2], rl[0], rl[3]-rl[2], rl[1]-rl[0])
        mbbox = (ml[2], ml[0], ml[3]-ml[2], ml[1]-ml[0])
    else:
        rbbox = (rl[0], rl[2], rl[1]-rl[0], rl[3]-rl[2])
        mbbox = (ml[0], ml[2], ml[1]-ml[0], ml[3]-ml[2])
    
    return rbbox, mbbox

def relative_locations(bbox, loc):
    corner = (bbox[0], bbox[1])
    size = (bbox[2], bbox[3])
    start_x = int(np.floor(corner[1] + loc[1]*size[1] ))
    start_y = int(np.floor(corner[0] + loc[0]*size[0]))
    
    end_x = int(np.ceil(start_x + loc[3]*size[1]))
    end_y = int(np.ceil(start_y + loc[2]*size[0]))
    
    return (start_y, end_y, start_x, end_x)

def scale_crop(image, scales):
    sx, sy = scales
    scale_max = max(np.ceil(sx), np.ceil(sy))
    image, image, mr, mm, loc = padding(image, 
                                        image,
                                        scaling=scale_max,
                                        dtype=np.uint8,
                                        mask=True,
                                        window=False
                                        )
    a = 0
    dx = 0
    dy = 0
    transform = [dx, dy, math.radians(a), sx, sy]
    image = warp(image, transform)
    mm = warp(mm, transform)
    bbox = compute_bbox(mm)
    print(bbox)
    implot(image)
    loc = bbox
    image = image[loc[1]:loc[1]+loc[3], loc[0]:loc[0]+loc[2]]

def local_median(image, k):
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            l = 3


def locally_normalize_gradients(image, complex=True, s=0.2, eta='mean'):
    rg1 = np.real(image)
    rg2 = np.imag(image)
    eta = 'mean'
    s = 0.01
    k = 51
    if eta == "median":
        obs = s * signal.medfilt2d(np.abs(image), k)
    elif eta == "mean":
        obs = s * ndimage.uniform_filter(np.abs(image), k)

    print("obs stats", np.min(obs), np.median(obs), np.max(obs), np.mean(obs))
    obs = np.clip(obs, 0.1, np.max(obs))

    #obs = 0.001#s*np.mean(image)
    #implot(obs)
    thres = 1
    #image = (image > thres)*image
    image = image / np.sqrt(rg1**2 + rg2**2 + obs**2)

    if complex:
        return image
    else:
        return np.real(image), np.imag(image)


def normalize_gradients(image, eta=None, complex=True, s=0.2, scales=None):
    rg1 = np.real(image)
    rg2 = np.imag(image)

    sw = 1
    sh = 1
    if scales is not None:
        sw, sh = scales
        rg1 = rg1*sw
        rg2 = rg2*sh

    if eta is None:
        eta = 0.001  #5*np.min(np.abs(image[np.abs(image)>0]))
    if eta is 'Mean':
        eta = s * np.mean(np.abs(image[np.abs(image) > 0]))

    #print('eta:', eta)
    norm = np.sqrt(rg1**2 + rg2**2 + eta**2)

    if complex and scales is None:
        return image / norm
    elif complex and scales is not None:
        return ( rg1 + 1j * rg2 ) / norm
    else:
        return rg1 / norm, rg2 / norm


def quiver_plot(gradient, title="Gradient Orientations"):
    fig, ax = plt.subplots()
    q = ax.quiver(gradient.real, gradient.imag)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()


def NNGF(reference, target, mask):
    reference = reference * mask
    reference = reference - np.mean(reference)
    reference = reference / max(np.var(target), 0.1)

    target = target - np.mean(target)
    target = target / np.var(target)

    implot(target)

    return spatialNGF(reference, target)


def spatialNGF(reference, target, normalize=False):
    #Assume both images are the same resolution and aligned.

    height, width = reference.shape

    pointwise = np.zeros((height, width))
    rg1 = reference.real
    rg2 = reference.imag

    tg1 = target.real
    tg2 = target.imag

    pointwise = (rg1 * tg1 + rg2 * tg2)**2

    if normalize:
        normalize = np.sum(np.abs(target) > 0)
    else:
        normalize = 1

    return np.sum(np.ravel(pointwise) / normalize), pointwise


def NCC(r, t):
    F_r = np.fft.fft2(r)
    F_t = np.fft.fft2(t)

    corr = F_r * F_t.conjugate()
    corr = np.fft.ifft2(corr)


def piecewiseNGF(reference, target, normalize=True):
    height, width = reference.shape

    rg1 = reference.real
    rg2 = reference.imag

    tg1 = target.real
    tg2 = target.imag

    if normalize:
        norm = np.sum(np.abs(target) > 0)
    else:
        norm = 1

    pointwise_real = (rg1**2 * tg1**2) / norm
    pointwise_imag = (rg2**2 * tg2**2) / norm
    pointwise_cross = 2 * (rg1 * rg2) * (tg1 * tg2) / norm

    return pointwise_real, pointwise_imag, pointwise_cross


def CC(reference, target):

    height, width = reference.shape

    pointwise = np.zeros((height, width))

    rg1 = reference.real
    rg2 = reference.imag
    tg1 = target.real
    tg2 = target.imag

    pointwise = (rg1 * tg1 + rg2 * tg2)

    normalize = 1  #np.sum(np.abs(target) > 0)

    return np.sum(np.ravel(pointwise) / normalize), pointwise


def C(reference, target):
    height, width = reference.shape
    pointwise = np.zeros((height, width), dtype=np.complex128)

    pointwise = (reference.conjugate() * target)
    normalize = 1  # np.sum(np.abs(target) > 0)
    return np.sum(np.ravel(pointwise) / normalize), pointwise


def power2(x):
    return 2**(math.ceil(math.log(x, 2)))


def padding(reference,
            target,
            scaling=1,
            dtype=np.complex64,
            bbox=False,
            mask=False,
            window=False):
    try:
        reference_height, reference_width, cr = reference.shape
    except ValueError:
        reference_height, reference_width = reference.shape
        cr = 1

    try:
        target_height, target_width, ct = target.shape
    except ValueError:
        target_height, target_width = target.shape
        ct = 1

    channels = max(cr, ct)

    if reference_height % 2 != 0:
        reference_height = reference_height - 1
    if reference_width % 2 != 0:
        reference_width = reference_width - 1

    if target_height % 2 != 0:
        target_height = target_height - 1
    if target_width % 2 != 0:
        target_width = target_width - 1

    Mh = int(scaling * max([reference_height, target_height]))
    Mw = int(scaling * max([reference_width, target_width]))

    Mw = power2(Mw)
    Mh = power2(Mh)
    #print("Padding to size: ", (Mh, Mw))

    reference = reference[:reference_height, :reference_width]
    target = target[:target_height, :target_width]

    #k = 3
    #const = (np.mean(reference[:k, :k]) + np.mean(reference[:k, -k:]) +np.mean(reference[-k:, :k]) + np.mean(reference[-k:, -k:]))//4

    new_reference = np.zeros((Mh, Mw, channels), dtype=dtype)
    height_offset = (Mh - reference_height) // 2
    width_offset = (Mw - reference_width) // 2

    #print(new_reference.shape)
    #print(reference.shape)
    #print("WO", width_offset)
    #print("HO", height_offset)

    new_reference[height_offset:height_offset + reference_height,
                  width_offset:width_offset +
                  reference_width, :cr] = reference.reshape(
                      (reference_height, reference_width, cr))
    #new_reference[height_offset:height_offset+reference_height, width_offset:width_offset+reference_width] = reference
    ref_bbox = (height_offset, height_offset + reference_height, width_offset,
                width_offset + reference_width)

    #k = 3
    #const = (np.mean(target[:k, :k]) + np.mean(target[:k, -k:]) +np.mean(target[-k:, :k]) + np.mean(target[-k:, -k:]))//4

    new_target = np.zeros((Mh, Mw, channels), dtype=dtype)
    height_offset = (Mh - target_height) // 2
    #print(Mh-target_height)
    #print(Mw-target_width)
    width_offset = (Mw - target_width) // 2
    #print("th", height_offset)
    new_target[height_offset:height_offset + target_height,
               width_offset:width_offset + target_width, :ct] = target.reshape(
                   (target_height, target_width, ct))
    #new_target[height_offset:height_offset+target_height, width_offset:width_offset+target_width] = target
    tar_bbox = (height_offset, height_offset + target_height, width_offset,
                width_offset + target_width)

    if bbox:
        return new_reference, new_target, ref_bbox, tar_bbox

    if channels == 1:
        new_reference = new_reference.reshape(new_reference.shape[0],
                                              new_reference.shape[1])
        new_target = new_target.reshape(new_target.shape[0],
                                        new_target.shape[1])

    if mask and channels == 1:
        k = 5 * 2
        reference_height = reference_height - k
        reference_width = reference_width - k
        mr = np.zeros((Mh, Mw), dtype=dtype)
        height_offset = (Mh - reference_height) // 2
        width_offset = (Mw - reference_width) // 2
        if window:
            mr[height_offset:height_offset + reference_height,
               width_offset:width_offset + reference_width] = create_window(
                   np.ones((reference_height, reference_width)))
        else:
            mr[height_offset:height_offset + reference_height,
               width_offset:width_offset + reference_width] = 1

        target_height = target_height - k
        target_width = target_width - k
        mt = np.zeros((Mh, Mw), dtype=dtype)

        height_offset = (Mh - target_height) // 2
        width_offset = (Mw - target_width) // 2

        #print("thm", height_offset)
        if window:
            mt[height_offset:height_offset + target_height,
               width_offset:width_offset + target_width] = create_window(
                   np.ones((target_height, target_width)))
        else:
            mt[height_offset:height_offset + target_height,
               width_offset:width_offset + target_width] = 1
        return new_reference, new_target, mr, mt, (ref_bbox, tar_bbox)

    return new_reference, new_target, (ref_bbox, tar_bbox)


"""
def padding(reference, target, scaling = 1, dtype=np.complex128, bbox=False):
    try:
        reference_height, reference_width, cr = reference.shape
    except ValueError:
        reference_height, reference_width = reference.shape
        cr = 1

    try:
        target_height, target_width, ct = target.shape
    except ValueError:
        target_height, target_width = target.shape
        ct = 1

    channels = max(cr, ct)

    if reference_height%2 != 0:
        reference_height = reference_height - 1
    if reference_width%2 != 0:
        reference_width = reference_width - 1

    if target_height%2 != 0:
        target_height = target_height - 1
    if target_width%2 != 0:
        target_width = target_width - 1

    Mh = int(scaling* max([reference_height, target_height]))
    Mw = int(scaling* max([reference_width, target_width]))

    Mw = power2(Mw)
    Mh = power2(Mh)
    #print("Padding to size: ", (Mh, Mw))


    reference = reference[:reference_height, :reference_width]
    target = target[:target_height, :target_width]

    #k = 3
    #const = (np.mean(reference[:k, :k]) + np.mean(reference[:k, -k:]) +np.mean(reference[-k:, :k]) + np.mean(reference[-k:, -k:]))//4

    new_reference = np.zeros((Mh, Mw, channels), dtype=dtype)
    height_offset = (Mh-reference_height)//2
    width_offset = (Mw-reference_width)//2

    #print(new_reference.shape)
    #print(reference.shape)
    #print("WO", width_offset)
    #print("HO", height_offset)

    new_reference[height_offset:height_offset+reference_height, width_offset:width_offset+reference_width, :cr] = reference.reshape((reference_height, reference_width, cr))
    #new_reference[height_offset:height_offset+reference_height, width_offset:width_offset+reference_width] = reference
    ref_bbox = (height_offset, height_offset+reference_height, width_offset, width_offset+reference_width)

    #k = 3
    #const = (np.mean(target[:k, :k]) + np.mean(target[:k, -k:]) +np.mean(target[-k:, :k]) + np.mean(target[-k:, -k:]))//4

    new_target = np.zeros((Mh, Mw, channels), dtype=dtype)
    height_offset = (Mh-target_height)//2
    width_offset = (Mw-target_width)//2

    new_target[height_offset:height_offset+target_height, width_offset:width_offset+target_width, :ct] = target.reshape((target_height, target_width, ct))
    #new_target[height_offset:height_offset+target_height, width_offset:width_offset+target_width] = target
    tar_bbox = (height_offset, height_offset+target_height, width_offset, width_offset+target_width)


    if bbox:
        return new_reference, new_target, ref_bbox, tar_bbox

    if channels == 1:
        new_reference = new_reference.reshape(new_reference.shape[0], new_reference.shape[1])
        new_target = new_target.reshape(new_target.shape[0], new_target.shape[1])

    return new_reference, new_target
"""


def highpass(shape):
    x = np.outer(np.cos(np.linspace(-math.pi / 2.0, math.pi / 2.0, shape[0])),
                 np.cos(np.linspace(-math.pi / 2.0, math.pi / 2.0, shape[1])))
    return (1.0 - x) * (2.0 - x)


def create_window(reference, plot=False, type="Tukey", alpha=0.05):
    height, width = reference.shape

    if type == "Hanning":
        hanning_window = cv2.createHanningWindow((width, height), cv2.CV_64F)
        #hanning_window = hanning_window[height//2, :]
        #hanning_window = np.outer(hanning_window, hanning_window)

        if plot:
            plt.contourf(hanning_window)
            plt.show()

        return hanning_window
    if type == "Tukey":
        a = alpha
        tukey_window_w = signal.tukey(width, sym=False, alpha=a)
        tukey_window_h = signal.tukey(height, sym=False, alpha=a)
        tukey_window = np.outer(tukey_window_h, tukey_window_w)
        return tukey_window


def centered_difference(image, axis=0):
    difference_kernel = [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
    n = len(difference_kernel)
    kernel = np.zeros((n, n), dtype=np.float32)
    difference_kernel = [1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]
    if axis == 0:
        kernel[1, :] = difference_kernel
    else:
        kernel[:, 1] = difference_kernel

    dd = cv2.CV_32F
    #dd = -1
    dst = cv2.filter2D(image, dd, kernel)
    return dst


def sobel(image, axis=0):
    if axis == 0:
        dst = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    else:
        dst = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return dst


def gradient(image):
    GR_x = centered_difference(image, axis=0)
    GR_y = centered_difference(image, axis=1)
    GR = GR_x + 1j * GR_y
    return GR


def derivative(reference, target, method='Difference'):

    if method == 'Sobel':
        GR_x = sobel(reference, axis=0)
        #GR_x = np.absolute(GR_x)

        GR_y = sobel(reference, axis=1)
        #GR_y = np.absolute(GR_y)

        GR = GR_x + 1j * GR_y

        GT_x = sobel(target, axis=0)
        #GT_x = np.absolute(GT_x)

        GT_y = sobel(target, axis=1)
        #GT_y = np.absolute(GT_y)

        GT = GT_x + 1j * GT_y
    elif method == 'Difference':
        GR_x = centered_difference(reference, axis=0)
        #GR_x = np.absolute(GR_x)

        GR_y = centered_difference(reference, axis=1)
        #GR_y = np.absolute(GR_y)

        GR = GR_x + 1j * GR_y

        GT_x = centered_difference(target, axis=0)
        #GT_x = np.absolute(GT_x)

        GT_y = centered_difference(target, axis=1)
        #GT_y = np.absolute(GT_y)

        GT = GT_x + 1j * GT_y

    return GR, GT


def fourier(image, scaling=1, plot=False, windowing=True, label=""):
    if windowing:
        window = create_window(image)
        image = image*window
    
    if plot:
        fig = plt.figure()
        plt.imshow(np.abs(image))
        plt.title("FFT Setup: "+str(label))
        plt.show()

    image_f = np.fft.fft2(image)

    if plot:
        fig = plt.figure()
        plt.imshow(np.log(np.abs(image_f)))
        plt.title("FFT After: "+str(label))
        plt.show()

    return image_f

def split_fourier(Y):
    Y = np.fft.fftshift(Y)
    Y_c = Y[::-1].conjugate()
    X1 = 0.5 * (Y_c + Y)
    X2 = 0.5j * (Y_c - Y)
    X1 = np.fft.fftshift(X1)[::-1]
    X2 = np.fft.fftshift(X2)[::-1]
    return X1, X2

def ft(reference, target, scaling=1, plot=False, windowing=True, label=""):

    if plot:
        fig = plt.figure()
        plt.imshow(reference.real)
        plt.title('FFT Setup: R (Before)' + str(label))
        plt.show()

    if windowing:
        window = create_window(reference)
        wref = reference * window

        window = create_window(target)
        wtar = target * window
    else:
        wref = reference
        wtar = target

    if plot:
        fig = plt.figure()
        plt.imshow(np.abs(wref))
        plt.title('FFT Setup: R (Window)' + str(label))
        plt.show()

    pref, ptar, loc = padding(wref, wtar, scaling=scaling)

    if plot:
        fig = plt.figure()
        plt.imshow(np.abs(pref))
        plt.title('FFT Setup: R' + str(label))
        plt.show()

        fig = plt.figure()
        plt.imshow(np.abs(ptar))
        plt.title('FFT Setup: T' + str(label))
        plt.show()

    R_f = np.fft.fft2(pref)
    T_f = np.fft.fft2(ptar)

    if plot:
        fig = plt.figure()
        plt.imshow(np.abs(R_f))
        plt.title('FFT Setup: R_f' + str(label))
        plt.show()

        fig = plt.figure()
        plt.imshow(np.abs(T_f))
        plt.title('FFT Setup: T_f' + str(label))
        plt.show()

    return R_f, T_f


def check_scan(rmask, tmask):

    r1, t1 = ft(rmask,
                tmask,
                scaling=1,
                plot=False,
                windowing=False,
                label="Mask")
    s = r1 * t1.conjugate()
    s = np.abs(np.fft.ifft2(s))
    return np.sum(np.abs(tmask) > 0), s

def MeanNGF1(reference,
                moving,
                plot=False,
                scaling=1,
                label="MeanNGF",
                windowing=False,
                masks=None,
                carea=1,
                scales=None):
    
    eta = 0.001
    
    mask_r, mask_m = masks
    sw, sh = scales
    
    reference = normalize_gradients(reference, eta=eta, scales=(1,1))
    moving = normalize_gradients(moving, eta=eta, scales=(1, 1))
    
    #subtract mean
    area_m = np.sum(np.abs(mask_m) > 0)
    moving = mask_m*(moving - (np.sum(moving)/area_m) )
    
    rg1 = np.real(reference)
    rg2 = np.imag(reference)
    sqr = rg1*rg1 + 1j * (rg2*rg2)
    
    mg1 = np.real(moving)
    mg2 = np.imag(moving)
    sqm = mg1*mg1 + 1j * (mg2*mg2)
    
    cross = rg1*rg2 + 1j * (mg1*mg2)
    
    #area_r = np.sum(np.abs(mask_r) > 0)
    #area = min(area_m, area_r)
    
    #Compute Squared
    fsqr, fsqm = ft(sqr,
            sqm,
            scaling=scaling,
            plot=False,
            windowing=windowing,
            label=" sqr real ")
    
    #Compute Cross and Reference
    fcross, fref = ft(cross,
            reference,
            scaling=scaling,
            plot=plot,
            windowing=windowing,
            label=" cross ")
    
    #Mask terms
    fmask_m, fmask_r = ft(mask_m, 
                          mask_r, 
                          scaling=scaling,
                          windowing=windowing,
                          label="")
    
    r1_2, r2_2 = split_fourier(fsqr)
    
    m1_2, m2_2 = split_fourier(fsqm)
    
    r1, r2 = split_fourier(fref)
    
    rc, mc = split_fourier(fcross)
    
    #m_int = np.fft.ifft2(fsqm * fmask_r.conjugate())
    #mc_int = np.fft.ifft2(mc * fmask_r.conjugate() )
    #m1_int = m_int.real
    #m2_int = m_int.imag
    
    
    #m1_int = np.sum(mg1*mg1)
    #m2_int = np.sum(mg2*mg2)
    #mc_int = np.sum(mg1*mg2)
    
    #t_area = np.fft.ifft2( fmask_r * fmask_m.conjugate() ).real + 1000
    t_area = 1#area_m
    
    #mean_r1 = 0#np.fft.ifft2( r1 * fmask_m.conjugate() ).real/t_area
    #mean_r2 = 0#np.fft.ifft2( r2 * fmask_m.conjugate() ).real/t_area
    #mean_r1_2 = np.fft.ifft2( r1_2 * fmask_m.conjugate() ).real/t_area
    #mean_r2_2 = np.fft.ifft2( r2_2 * fmask_m.conjugate() ).real/t_area
    
    fcorr_g1 = r1_2 * m1_2.conjugate() 
    fcorr_g2 = r2_2 * m2_2.conjugate() 
    fcorr_mix = rc * mc.conjugate()
    
    fcorr_ngf_base = fcorr_g1 + fcorr_g2 + 2*fcorr_mix
    corr_ngf_base = np.fft.ifft2(fcorr_ngf_base)
    #implot(np.fft.fftshift(corr_ngf_base))
    
    #M_1^2R_1
    #fcorr_m1_2_r1 =  m1_2 * r2.conjugate() 
    #M_2^2R_2
    #fcorr_m2_2_r2 = m2_2 * r2.conjugate()
    
    #corr_m1_2_r = np.fft.ifft2(fcorr_m1_2_r1 + 1j * fcorr_m2_2_r2)
    #corr_m1_2_r1 = corr_m1_2_r.real
    #corr_m2_2_r2 = corr_m1_2_r.real
    
    #M_cross R_1
    #fcorr_mc_r1 = mc * r1.conjugate() 
    #M_cross R_2
    #fcorr_mc_r2 = mc * r2.conjugate() 

    #corr_mc_r = np.fft.ifft2(fcorr_mc_r1 + 1j * fcorr_mc_r2)
    #corr_mc_r1 = corr_mc_r.real
    #corr_mc_r2 = corr_mc_r.imag

    #put it all together
    corr = corr_ngf_base \
            + mean_r1 * ( mean_r1 * m1_int + mean_r2 * mc_int - 2 * corr_m1_2_r1 - corr_mc_r1 ) \
            + mean_r2 * ( mean_r2 * m2_int - 2 * corr_m2_2_r2 - corr_mc_r2 )

    corr = corr#/t_area
    implot(np.fft.fftshift(corr))
    
    Dy, Dx = np.unravel_index(corr.argmax(), corr.shape)
    height, width = np.shape(corr)
    peak = corr[Dy, Dx]

    if plot:
        fig = plt.figure()
        plt.imshow(np.log(corr + 0.02))
        plt.title(label + ': Correlation MeanNGF')
        plt.show()

    if (Dy > height // 2):
        Dy -= height

    if (Dx > width // 2):
        Dx -= width

    return [Dy, Dx], peak, corr


def MeanNGF(reference, 
         moving, 
         plot=False,
         scaling=1,
         label="MeanNGF",
         windowing=False,
         masks=None,
         carea = 1,
         scales=None):
    
    eta = 0.01
    
    mask_r, mask_m = masks
    sw, sh = scales
    
    reference = normalize_gradients(reference, eta=eta, scales=(1, 1))
   
    moving = normalize_gradients(moving, eta=eta, scales=scales)
    
    moving = mask_m*(moving - (np.sum(mask_m*moving)/np.sum(mask_m) ))
    denom_m = np.sum(moving.real**2 + moving.imag**2)
    
    rg1 = np.real(reference)
    rg2 = np.imag(reference)
    
    mg1 = np.real(moving)
    mg2 = np.imag(moving)

    area_m = np.sum(np.abs(mask_m) > 0)
    area_r = np.sum(np.abs(mask_r) > 0)
    area = min(area_m, area_r)
    
    #Real Part FFT (squared)
    r1_2, m1_2 = ft(rg1 * rg1,
                mg1 * mg1,
                scaling=scaling,
                plot=False,
                windowing=windowing,
                label=" sqr real ")

    #Imaginary Part FFT (squared)
    r2_2, m2_2 = ft(rg2 * rg2,
                mg2 * mg2,
                scaling=scaling,
                plot=plot,
                windowing=windowing,
                label=" sqr imag ")

    #Cross Terms
    rc, mc = ft(rg1 * rg2,
                mg1 * mg2,
                scaling=scaling,
                plot=plot,
                windowing=windowing,
                label=" cross ")
    
    #Real terms for reference
    r1, r2 = ft(rg1,
                rg2,
                scaling=scaling,
                windowing=windowing,
                label="ref (not squared)")
    
    m1, m2 = ft(mg1,
                mg2,
                scaling-scaling,
                windowing=windowing,
                lable="mov (not squared)")
    
    #Mask terms
    fmask_m, fmask_r = ft(mask_m, 
                          mask_r, 
                          scaling=scaling,
                          windowing=windowing,
                          label="")
    
    #m1_int = np.sum(mg1*mg1)
    #m2_int = np.sum(mg2*mg2)
    #mc_int = np.sum(mg1*mg2)
    
    combined_area = np.fft.ifft2( fmask_r * fmask_m.conjugate() ).real # +1000
    #combined_area = np.clip(combined_area, a_min=1000, a_max=None)
    #print(area_m)
    #print(np.max(combined_area))
    """
    mean_r1 = np.fft.ifft2( r1 * fmask_m.conjugate() )/area_m
    mean_r2 = np.fft.ifft2( r2 * fmask_m.conjugate() )/area_m
    
    mean_r1_2 = np.fft.ifft2( r1_2 * fmask_m.conjugate() )/area_m
    mean_r2_2 = np.fft.ifft2( r2_2 * fmask_m.conjugate() )/area_m
    
    #implot(mean_r1, "Mean r1")
    #print(mean_r1)
    #print(np.max(mean_r1))
    
    corr_g1 = r1_2 * m1_2.conjugate()  #/ (np.abs(r1) * np.abs(t1.conjugate()))
    corr_g2 = r2_2 * m2_2.conjugate()  #/ (np.abs(r2) * np.abs(t2.conjugate()))
    corr_mix = rc * mc.conjugate()  #/ (np.abs(cr) * np.abs(ct.conjugate()))
    corr_ngf_base = np.fft.ifft2(corr_g1 + corr_g2 + 2*corr_mix)
    
    #M_1^2R_1
    corr_m1_2_r1 = np.fft.ifft2( m1_2 * r2.conjugate() )#.real
    
    #M_2^2R_2
    corr_m2_2_r2 = np.fft.ifft2( m2_2 * r2.conjugate() )#.real
    
    #M_cross R_1
    corr_mc_r1 = np.fft.ifft2( mc * r1.conjugate() )#.real
    
    #M_cross R_2
    corr_mc_r2 = np.fft.ifft2( mc * r2.conjugate() )#.real


    #denominator - r
    denom_r = np.clip( (-mean_r1*mean_r1 - mean_r2*mean_r2+mean_r1_2+ mean_r2_2) * (combined_area > area_m-1) , a_min=0.01, a_max=None)
    denom_r = denom_r
    
    corr_mean_r1 = mean_r1 * ( mean_r1 * m1_int + mean_r2 * mc_int - 2 * corr_m1_2_r1 - corr_mc_r1 )
    corr_mean_r2 = mean_r2 * ( mean_r2 * m2_int - 2 * corr_m2_2_r2 - corr_mc_r2 )

    """
    
    corr_gx = np.fft.ifft(r1 * m1.conjugate())
    corr_gy = np.fft.ifft(r2 * m2.conjufate())
    
    #put it all together
    
            
    corr = corr_gx + corr_gy
    
    #print(denom_r)
    #print(denom_m)
    #implot(denom_r, "denom_r")
    #print("maxr", np.max(denom_r))
    #print("minr", np.min(denom_r))
    corr = corr/area_m
    #corr = corr/(denom_r*denom_m)
    #implot(np.fft.fftshift(corr), "before")
    #corr = corr * (combined_area > area_m-1)
    
    
    #implot(np.fft.fftshift(corr), "after")
    #implot(np.fft.fftshift(corr_mean_r1))
    #implot(np.fft.fftshift(corr_mean_r2))
    #print(np.max(corr_ngf_base))
    #print(np.max(corr_mean_r1))
    #print(np.max(corr_mean_r2))
    
    #corr = corr/combined_area
   
    Dy, Dx = np.unravel_index(corr.argmax(), corr.shape)
    height, width = np.shape(corr)
    peak = corr[Dy, Dx]

    if plot:
        fig = plt.figure()
        plt.imshow(np.log(corr + 0.02))
        plt.title(label + ': Correlation CC')
        plt.show()

    if (Dy > height // 2):
        Dy -= height

    if (Dx > width // 2):
        Dx -= width

    return [Dy, Dx], peak, corr

    
    
def NGF(reference,
        moving,
        plot=False,
        scaling=1,
        label="NGF",
        windowing=False,
        masks=None,
        carea = 1,
        scales=None
        ):
    #eta = 'Mean'
    eta = 0.01
    if masks is not None:
        mask_r, mask_m = masks
    else:
        mask_r = None
        mask_m = None

    sw, sh = scales
    print(sw, sh)
    
    reference = normalize_gradients(reference, eta=eta, scales=(1,1))
    #reference = locally_normalize_gradients(reference)
    rg1 = np.real(reference)
    rg2 = np.imag(reference)

    #target = target - np.mean(target[np.abs(target) > 0])
    #target = target / np.var(target[np.abs(target) > 0])

    moving = normalize_gradients(moving, eta=eta, scales=(1, 1))
    #moving = locally_normalize_gradients(moving)

    mg1 = np.real(moving)#*sw
    mg2 = np.imag(moving)#*sh

    area_m = np.sum(np.abs(mask_m) > 0)
    area_r = np.sum(np.abs(mask_r) > 0)
    area = min(area_m, area_r)
    print("Area: ", area_m, area_r, area, flush=True)

    #A, B = check_scan(rmask, tmask)
    #Real Part FFT (squared)
    r1, t1 = ft(rg1 * rg1,
                mg1 * mg1,
                scaling=scaling,
                plot=False,
                windowing=windowing,
                label=" sqr real ")

    #Imaginary Part FFT (squared)
    r2, t2 = ft(rg2 * rg2,
                mg2 * mg2,
                scaling=scaling,
                plot=plot,
                windowing=windowing,
                label=" sqr imag ")

    #Cross Terms
    cr, ct = ft(rg1 * rg2,
                mg1 * mg2,
                scaling=scaling,
                plot=plot,
                windowing=windowing,
                label=" cross ")

    #print(r1.dtype)
    corr_g1 = r1 * t1.conjugate()  #/ (np.abs(r1) * np.abs(t1.conjugate()))
    corr_g2 = r2 * t2.conjugate()  #/ (np.abs(r2) * np.abs(t2.conjugate()))
    corr_mix = cr * ct.conjugate()  #/ (np.abs(cr) * np.abs(ct.conjugate()))


    if scales is not None and False:
        print("Using Scale")
        sw, sh = scales
        corr_g1 = np.fft.ifft2(corr_g1)/(sh*sh)
        corr_g2 = np.fft.ifft2(corr_g2)/(sw*sw)
        corr_mix = np.fft.ifft2(corr_mix)/(sw*sh)
        corr = corr_g1 + corr_g2 + 2*corr_mix
    else:
        sw, sh = scales
        corr = np.fft.ifft2((corr_g1 + corr_g2 + 2 * corr_mix))  #/(sumc))
        corr = corr / area
        
    Dy, Dx = np.unravel_index(corr.argmax(), corr.shape)
    height, width = np.shape(corr)
    peak = corr[Dy, Dx]

    if plot:
        fig = plt.figure()
        plt.imshow(np.log(corr + 0.02))
        plt.title(label + ': Correlation CC')
        plt.show()

    if (Dy > height // 2):
        Dy -= height

    if (Dx > width // 2):
        Dx -= width

    return [Dy, Dx], peak, corr


def entropy(hist):
    hist = hist / float(np.sum(hist))
    hist = hist[np.nonzero(hist)]
    return -np.sum(hist * np.log2(hist))


def get_segment_crop(img, tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def max2d(data):
    Dy, Dx = np.unravel_index(data.argmax(), data.shape)
    return (Dy, Dx)

def mutual_information(reference, moving, bins=256, normed=True, mask=None):
    if mask is not None:
        reference = reference * mask

    ref_range = (reference.min(), reference.max())
    mov_range = (moving.min(), moving.max())

    joint_hist, _, _ = np.histogram2d(reference.flatten(),
                                      moving.flatten(),
                                      bins=bins,
                                      range=[ref_range, mov_range])
    ref_hist, _ = np.histogram(reference, bins=bins, range=ref_range)
    mov_hist, _ = np.histogram(moving, bins=bins, range=mov_range)
    joint_ent = entropy(joint_hist)
    ref_ent = entropy(ref_hist)
    mov_ent = entropy(mov_hist)
    mutual_info = ref_ent + mov_ent - joint_ent

    if normed:
        mutual_info = mutual_info / np.sqrt(ref_ent * mov_ent)

    return mutual_info


def normed_cross_correlation(reference,
                      moving, 
                      plot=False,
                      scaling=1,
                      label="CC",
                      masks=None,
                      windowing=False):
    
    height, width = reference.shape

    rm, tm = masks
    box = compute_bbox(moving)
    I = box_to_crop(box, moving, reverse=True)
    corr = cv2.matchTemplate(reference, I, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
    corr = np.fft.fftshift(corr)
    
    
    try:
        Dy, Dx = np.unravel_index(corr.argmax(), corr.shape)
    except ValueError:
        Dy, Dx, c = np.unravel_index(corr.argmax(), corr.shape)

    peak = max_val

    corner = max_loc
    diff = [0, 0]
    diff[1] = box[1]-corner[1]
    diff[0] = box[0]-corner[0]

    #implot(corr)
    #peak = peak/area

    if (Dy > height // 2):
        Dy -= height

    if (Dx > width // 2):
        Dx -= width

    return [-1* diff[1], -1*diff[0]], peak, corr


def cross_correlation(reference,
                      moving, 
                      plot=False,
                      scaling=1,
                      label="CC",
                      masks=None,
                      windowing=False,
                      scales=None,
                      carea=None,
                      mean=True,
                      std=True):
    
    height, width = reference.shape
    sw, sh = scales
    
    #Gradient Angle Invariance
    """
    a = np.angle(reference)
    reference = (a < 0)*reference*np.exp(1j*np.pi) + (a >= 0)*reference

    a = np.angle(moving)
    target = (a < 0)*moving*np.exp(1j*np.pi) + (a >= 0)*moving
    """
    
    #rg1 = reference.real# / sw
    #rg2 = reference.imag# / sh
    #reference = rg1 + 1j*rg2
    
    rg1 = moving.real * sw
    rg2 = moving.imag * sh
    moving = rg1 + 1j*rg2
    rm, mm = masks
    #print(np.sum(moving))
    if mean:
        moving = mm*(moving - (np.sum(mm*moving)/np.sum(mm) ))
    denom_m = np.sum(moving.real**2 + moving.imag**2)
    
    #print(np.sum(moving))
    
    R_f, M_f = ft(reference,
              moving,
              scaling=scaling,
              plot=plot,
              windowing=windowing)

    rg1 = reference.real
    rg2 = reference.imag
    
    r1, r2 = ft(rg1, rg2, scaling=scaling, windowing=windowing)
    
    if std:
        r1_2, r2_2 = ft(rg1*rg1, rg2*rg2, scaling=scaling, windowing=windowing)
    
    if mean:
        fmask_m, fmask_r = ft(mm, rm, scaling=scaling, windowing=windowing)
    
    #R_f = fourier(reference, scaling=scaling, plot=plot, windowing=windowing)
    #M_f = fourier(moving, scaling=scaling, plot=plot, windowing=windowing)
    
    corr_f = R_f * M_f.conjugate()
    corr = np.fft.ifft2(corr_f)
    
    area_m = np.sum(np.abs(mm) > 0)
    area_r = np.sum(np.abs(rm) > 0)
    area = min(area_m, area_r)
    
    if mean:
        mean_r1 = np.fft.ifft2( r1 * fmask_m.conjugate() )/area_m
        mean_r2 = np.fft.ifft2( r2 * fmask_m.conjugate() )/area_m
    
    if std:
        mean_r1_2 = np.fft.ifft2( r1_2 * fmask_m.conjugate() )/area_m
        mean_r2_2 = np.fft.ifft2( r2_2 * fmask_m.conjugate() )/area_m
    
        combined_area = np.fft.ifft2( fmask_r * fmask_m.conjugate() ).real 
    
    if std:
        denom_r = np.clip( (-mean_r1*mean_r1 - mean_r2*mean_r2+mean_r1_2+ mean_r2_2) * (combined_area > area_m-1) , a_min=0.01, a_max=None)
        denom_r = denom_r
    
    if std:
        corr = corr/ (np.sqrt(denom_r)*np.sqrt(denom_m) )
    else:
        corr = corr/ area
        
    if std:
        corr = corr * (combined_area > area_m-1)
    
    corr = corr.real
    try:
        Dy, Dx = np.unravel_index(corr.argmax(), corr.shape)
    except ValueError:
        Dy, Dx, c = np.unravel_index(corr.argmax(), corr.shape)

    peak = corr[Dy, Dx]

    if (Dy > height // 2):
        Dy -= height

    if (Dx > width // 2):
        Dx -= width

    return [Dy, Dx], peak, corr

def phase_correlation(reference,
                      moving,
                      plot=False,
                      scaling=1,
                      label="Phase Correlation",
                      masks=None,
                      windowing=False,
                      filter=False):
    #Normalization
    """
    eta = 0.1
    g1 = np.real(reference)
    g2 = np.imag(reference)
    norm = np.sqrt(g1**2 + g2**2 + eta)
    reference = reference / norm

    g1 = np.real(target)
    g2 = np.imag(target)
    norm = np.sqrt(g1**2 + g2**2 + eta)
    target = target / norm
    """

    #Gradient Angle Invariance
    
    a = np.angle(reference)
    reference = (a < 0)*reference*np.exp(1j*np.pi) + (a >= 0)*reference

    a = np.angle(moving)
    moving = (a < 0)*moving*np.exp(1j*np.pi) + (a >= 0)*moving
    
    if filter:
        reference = cv2.GaussianBlur(reference, (3, 3), 0)
        moving= cv2.GaussianBlur(moving, (3, 3), 0)

    R_f, M_f = ft(reference,
                  moving,
                  scaling=scaling,
                  plot=plot,
                  windowing=windowing)

    #R_f = fourier(reference, scaling=scaling, plot=plot, windowing=windowing)
    #M_f = fourier(moving, scaling=scaling, plot=plot, windowing=windowing)
    
    try:
        height, width, channels = R_f.shape
    except ValueError:
        height, width = R_f.shape

    eps = 0.001  #1e-10
    thres = 0.01

    corr_f = R_f * M_f.conjugate()
    corr_f /= (np.absolute(R_f) * np.absolute(M_f.conjugate()) + eps)
    corr_f = corr_f * (np.absolute(R_f) > thres) * (np.absolute(M_f.conjugate()) > thres)
    corr = np.fft.ifft2(corr_f)

    try:
        Dy, Dx = np.unravel_index(corr.argmax(), corr.shape)
    except ValueError:
        Dy, Dx, c = np.unravel_index(corr.argmax(), corr.shape)

    peak = corr[Dy, Dx]

    if plot:
        fig = plt.figure()
        plt.imshow(np.log(np.abs(R_f)))
        plt.title(label + ': FFT R')
        plt.show()

        fig = plt.figure()
        plt.imshow(np.log(np.abs(R_f)))
        plt.title(label + ': FFT R')
        plt.show()

        fig = plt.figure()
        plt.semilogy(np.abs(R_f[height // 2, :]))
        plt.title(label + ': FFT T')
        plt.show()

        fig = plt.figure()
        plt.imshow(np.log(corr + 0.02))
        plt.title(label + ': Correlation')
        plt.show()

    if (Dy > height // 2):
        Dy -= height

    if (Dx > width // 2):
        Dx -= width

    return [Dy, Dx], peak, corr


def loglog(image, ovrsamp=None, res=(None, None), log_base=(None, None)):
    height, width, channels = image.shape

    res_h = res[0]
    res_w = res[1]

    log_base_h = log_base[0]
    log_base_w = log_base[1]

    if ovrsamp is None:
        ovrsamp = 1

    if res_h is None:
        res_h = height * ovrsamp

    if res_w is None:
        res_w = width * ovrsamp

    if log_base_h is None:
        mr = np.log10(height / 2) / res_h
        log_base_h = 10**(mr)

    if log_base_w is None:
        mr = np.log10(width / 2) / res_w
        log_base_w = 10**(mr)

    lh = np.empty(res_h)
    lh[:] = np.power(log_base_h,
                     np.linspace(0, res_h, num=res_h, endpoint=False))

    lw = np.empty(res_w)
    lw[:] = np.power(log_base_w,
                     np.linspace(0, res_w, num=res_w, endpoint=False))

    eh = height // 2
    oh = np.linspace(0, eh, num=eh, endpoint=False)

    ew = width // 2
    ow = np.linspace(0, ew, num=ew, endpoint=False)
    #print(ow)

    #image = np.flip(image[:eh, :ew])
    image = image[:eh, :ew]

    of = interpolate.interp2d(oh, ow, image, kind='linear')
    log_interp = of(lh, lw)

    #log_interp = ndii.map_coordinates(image, [lh, lw])
    #print(image.shape)
    #print(log_interp.shape)

    return log_interp, (log_base_h, log_base_w)


def poc2warp(center, param):
    try:
        cx, cy, c = center
    except ValueError:
        cx, cy = center

    dx, dy, theta, scalex, scaley = param
    cs = math.cos(theta)
    sn = math.sin(theta)

    Rot = np.float32([[scalex * cs, scaley * sn, 0],
                      [-scalex * sn, scaley * cs, 0], [0, 0, 1]])
    center_Trans = np.float32([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    center_iTrans = np.float32([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    cRot = np.dot(np.dot(center_Trans, Rot), center_iTrans)
    Trans = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    Affine = np.dot(cRot, Trans)

    return Affine

def warp_point(point, params, shape, inverse=False):
    center = np.array(shape)/2
    center = np.flip(center)
    
    x, y = point
    array = np.array([x, y, 1])
    
    angle = params[0]
    sw = params[1]
    sh = params[2]
    dy = params[3]
    dx = params[4]

    if inverse:
        p = [-dy, -dx, 0, 1, 1]
        Affine = poc2warp(center, p)
        array = Affine @ array
        
        p = [0, 0, 0, 1/sw, 1/sh]
        Affine = poc2warp(center, p)
        array = Affine @ array
        
        p = [0, 0, -angle, 1, 1]
        Affine = poc2warp(center, p)
        array = Affine @ array
        
    else:
        array = array.T
        print(dy, dx, angle, sw, sh)
        p = [0, 0, angle, 1, 1]
        Affine = poc2warp(center, p)
        array = Affine @ array
        
        p = [0, 0, 0, sw, sh]
        Affine = poc2warp(center, p)
        array = Affine @ array 
        
        p = [dy, dx, 0, 1, 1]
        Affine = poc2warp(center, p)
        array = Affine @ array
    

    return (array[0], array[1])

def warp(Img, param):
    center = np.array(Img.shape) / 2
    center = np.flip(center)
    try:
        rows, cols, channels = Img.shape
    except:
        rows, cols = Img.shape
        channels = 1

    Affine = poc2warp(center, param)
    #k = 3
    #const = (np.mean(Img[:k, :k]) + np.mean(Img[:k, -k:]) +np.mean(Img[-k:, :k]) + np.mean(Img[-k:, -k:]))//4
    const = 0
    outImg = cv2.warpPerspective(Img,
                                 Affine, (cols, rows),
                                 cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=const)
    return outImg

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", time=None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if time is not None:
        work_left = float(total) - float(iteration)
        time_left = work_left*time
        
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if time is not None:
        print(f'\r{prefix} |{bar}| {percent}% {suffix} | {time_left} (s)', end = printEnd)
    else:
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


class Result:
    def __init__(self, peak, params, translation_surface=None):
        self.sim = peak
        self.params = params
        self.surface = translation_surface
        self.image = None
        self.mask = None
        self.ref = None
        
    def __repr__(self):
        return f"Parameter {self.params} -- Similarity: {self.sim}"

    def __lt__(self, other):
        return self.sim < other.sim
    
    def __eq__(self, other):
        return self.sim == other.sim

class MaxHeap():
    def __init__(self, top_n):
        self.h = []
        self.length = top_n
        heapify(self.h)

    def add(self, element):
        if len(self.h) < self.length:
            heappush(self.h, element)
        else:
            heappushpop(self.h, element)

    def getList(self):
        return nlargest(self.length, self.h)

def runModifiedSearch(reference,
              moving,
              params,
              masks=None,
              record=True,
              search_type="GPC", 
              top_k = 5, 
              progress=True
              ):

    if masks is not None:
        mask_r, mask_m = masks
    else:
        mask_r = None
        mask_m = None

    implot(reference, "reference")
    theta = np.linspace(params[0], params[1], (int)(params[2]))
    scale_w = np.linspace(params[3], params[4], (int)(params[5]))
    scale_h = np.linspace(params[6], params[7], (int)(params[8]))

    if record:
        surface = np.zeros((len(theta), len(scale_w), len(scale_h)))
    else:
        surface = None
    
    max_heap = MaxHeap(top_k)

    work = len(theta)*len(scale_w)*len(scale_h)
    
    sum_time = 0
    average_time = 0
    count = 0
    
    rbbox = compute_bbox(mask_r, quad=False)
    mbbox = compute_bbox(mask_m)
    rarea = (rbbox[2]-1)*(rbbox[3]-1)
    marea = (mbbox[2]-1)*(mbbox[3]-1)
    
    for i in range(len(theta)):
        for j in range(len(scale_w)):
            
            sw = scale_w[j]
            if sw <= 1:
                #Upsample Reference
                tm = moving.copy()
                tmask_m = mask_m
                transformation = [0, 0, 0, 1/sw, 1]
                tr = warp(reference.copy(), transformation)
                tmask_r = warp(mask_r.copy(), transformation)
                implot(tr, str(sw))
            else:
                print("less", sw)
                tr = reference.copy()
                tmask_r = mask_r.copy()
                transformation = [0, 0, 0, sw, 1]
                tm = warp(moving.copy(), transformation)
                tmask_m = warp(mask_m.copy(), transformation)
                
            time_chunk_start = time.perf_counter()
            
            if progress:
                printProgressBar(len(scale_h)*(i+1)*(j+1), work, prefix = 'Search Progress:', suffix = '', length = 50)
            
            
            for k in range(len(scale_h)):

                angle = theta[i]
                sh = scale_h[k]
                print("Eval:", sw, sh)
                if sh <= 1:
                    #Upsample Reference
                    transformation = [0, 0, 0, 1, 1/sh]
                    tr = warp(tr, transformation)
                    tmask_r = warp(tmask_r, transformation)
                else:
                    print("less")
                    transformation = [0, 0, 0, 1, sh]
                    tm = warp(tm , transformation)
                    tmask_m = warp(tmask_m, transformation)

                #Perform rotation
                transformation = [0, 0, angle, 1, 1]
                tm = warp(tm, transformation)
                tmask_m = warp(tmask_m , transformation)
                
                implot(tm, str(sw)+","+str(sh))
                implot(tr,  str(sw)+","+str(sh))
                #Compute Gradients
                td = gradient(tm)
                rd = gradient(tr)

                td = td*tmask_m
                rd = rd*tmask_r
                
                area = min(rarea, sw*sh*marea)
                print("Matching Area: ", area)

                if search_type is "NGF":
                    dx, peak, corr = NGF(rd,
                                         td,
                                         plot=False,
                                         scaling=1,
                                         windowing=False,
                                         masks=(tmask_r, tmask_m),
                                         scales=(sw, sh),
                                         carea=area
                                    )
                elif search_type is "GPC":
                    dx, peak, corr = phase_correlation(rd,
                                                       td,
                                                       plot=False,
                                                       scaling=1,
                                                       windowing=False,
                                                       masks=(tmask_r, tmask_m)
                                                    )
                elif search_type is "CCdif":
                    dx, peak, corr = cross_correlation(rd,
                                                       td, 
                                                       plot=False,
                                                       windowing=False,
                                                       masks=(rmask_r, tmask_m)
                                                    )
                elif search_type is "CC":
                    dx, peak, corr = cross_correlation(reference,
                                                       tn, 
                                                       plot=False,
                                                       windowing=False,
                                                       masks=(tmask_r, tmask_m)
                                                    )
                else:
                    try:
                        raise ValueError("Invalude Selection for Search Type")
                    except ValueError:
                        print("The supported options are: GPC (Phase Correlation) and NGF (Normalized Gradient Fields).")

                if record:
                    surface[i, j, k] = np.abs(peak)#.real

                result = Result(peak, (angle, sw, sh, dx[0], dx[1]), corr)
                max_heap.add(result)
                
            time_chunk_end = time.perf_counter()
            time_chunk = (time_chunk_end - time_chunk_start)/len(scale_h)
            count += 1
            sum_time += time_chunk
            average_time = time_chunk/count
            
    output = max_heap.getList()
    return output, surface




def runSearch(reference,
              moving,
              params,
              masks=None,
              record=True,
              search_type="GPC", 
              top_k = 5, 
              progress=True
              ):

    if masks is not None:
        mask_r, mask_m = masks
    else:
        mask_r = None
        mask_m = None

    theta = np.linspace(params[0], params[1], (int)(params[2]))
    scale_w = np.linspace(params[3], params[4], (int)(params[5]))
    scale_h = np.linspace(params[6], params[7], (int)(params[8]))

    if record:
        surface = np.zeros((len(theta), len(scale_w), len(scale_h)))
    else:
        surface = None
    
    max_heap = MaxHeap(top_k)
    
    #rd = reference + 1j * reference
    rd = gradient(reference)
    if mask_r is not None:
        rd = rd * mask_r

    work = len(theta)*len(scale_w)*len(scale_h)
    
    sum_time = 0
    average_time = 0
    count = 0
    
    rbbox = compute_bbox(mask_r, quad=False)
    mbbox = compute_bbox(mask_m)
    rarea = (rbbox[2]-1)*(rbbox[3]-1)
    marea = (mbbox[2]-1)*(mbbox[3]-1)
    
    for i in range(len(theta)):
        for j in range(len(scale_w)):
            
            time_chunk_start = time.perf_counter()
            
            if progress:
                printProgressBar(len(scale_h)*(i+1)*(j+1), work, prefix = 'Search Progress:', suffix = '', length = 50)
            
            
            for k in range(len(scale_h)):

                angle = theta[i]
                sw = scale_w[j]
                sh = scale_h[k]

                transformation = [0, 0, angle, sw, sh]

                tn = warp(moving, transformation)
                td = gradient(tn)

                area = min(rarea, sw*sh*marea)
                area = sw*sh#*marea
                print("Matching Area: ", area, rarea)
                if mask_m is not None:
                    transformed_mm = warp(mask_m, transformation)
                    td = td * transformed_mm
                else:
                    transformed_mm = None

                if search_type is "NGF":
                    dx, peak, corr = NGF(rd,
                                                         td,
                                                         plot=False,
                                                         scaling=1,
                                                         windowing=False,
                                                         masks=(mask_r, transformed_mm),
                                                         scales=(sw, sh),
                                                         carea=area
                                                    )
                elif search_type is "MeanNGF":
                    dx, peak, corr = MeanNGF(rd,
                                            td,
                                            plot=False,
                                            scaling=1,
                                            windowing=False,
                                            masks=(mask_r, transformed_mm),
                                            scales=(sw, sh),
                                            carea=area
                                            )
                    #implot(np.fft.fftshift(corr))
                elif search_type is "GPC":
                    dx, peak, corr = phase_correlation(rd,
                                                       td,
                                                       plot=False,
                                                       scaling=1,
                                                       windowing=False,
                                                       masks=(mask_r, transformed_mm)
                                                    )
                elif search_type is "GCC":
                    dx, peak, corr = cross_correlation(rd,
                                                       td, 
                                                       plot=False,
                                                       windowing=False,
                                                       masks=(mask_r, transformed_mm),
                                                       carea=area,
                                                       scales=(sw, sh),
                                                       mean=False,
                                                       std=False
                                                    )
                elif search_type is "GCC_mean":
                    dx, peak, corr = cross_correlation(rd,
                                   td, 
                                   plot=False,
                                   windowing=False,
                                   masks=(mask_r, transformed_mm),
                                   carea=area,
                                   scales=(sw, sh),
                                   mean=True,
                                   std=False
                                )
                elif search_type is "GCC_std":
                    dx, peak, corr = cross_correlation(rd,
                                   td, 
                                   plot=False,
                                   windowing=False,
                                   masks=(mask_r, transformed_mm),
                                   carea=area,
                                   scales=(sw, sh),
                                   mean=True,
                                   std=True
                                )                    
                elif search_type is "CC":
                    dx, peak, corr = cross_correlation(mask_r*reference,
                                                       transformed_mm*tn, 
                                                       plot=False,
                                                       windowing=False,
                                                       masks=(mask_r, transformed_mm),
                                                       carea=area,
                                                       scales=(1, 1),
                                                       mean=False,
                                                       std=False,
                                                    )
                elif search_type is "CC_mean":
                    dx, peak, corr = cross_correlation(mask_r*reference,
                                   transformed_mm*tn, 
                                   plot=False,
                                   windowing=False,
                                   masks=(mask_r, transformed_mm),
                                   carea=area,
                                   scales=(1, 1),
                                   mean=True,
                                   std=False,
                                )
                elif search_type is "NCC":
                    dx, peak, corr = normed_cross_correlation(
                                   mask_r*reference,
                                   transformed_mm*tn, 
                                   plot=False,
                                   windowing=False,
                                   masks=(mask_r, transformed_mm)
                                )
                else:
                    try:
                        raise ValueError("Invalude Selection for Search Type")
                    except ValueError:
                        print("The supported options are: GPC (Phase Correlation) and NGF (Normalized Gradient Fields).")

                if record:
                    surface[i, j, k] = np.abs(peak)#.real

                result = Result(peak, (angle, sw, sh, dx[0], dx[1]), corr)
                result.corr = corr
                max_heap.add(result)
                
            time_chunk_end = time.perf_counter()
            time_chunk = (time_chunk_end - time_chunk_start)/len(scale_h)
            count += 1
            sum_time += time_chunk
            average_time = time_chunk/count
            
    output = max_heap.getList()
    return output, surface


def multilevel_reduce(image, target_width):
    s = 0.2
    while (image.shape[0] * s > target_width):
        new_target_width = int(image.shape[0] * s)
        new_target_height = int(image.shape[1] * s)
        new_target_dim = (new_target_height, new_target_width)
        image = cv2.resize(image, new_target_dim, interpolation=cv2.INTER_AREA)

    s = 1 / (image.shape[0] / target_width)
    new_target_width = int(image.shape[0] * s)
    new_target_height = int(image.shape[1] * s)
    new_target_dim = (new_target_height, new_target_width)
    image = cv2.resize(image, new_target_dim, interpolation=cv2.INTER_AREA)
    return image


def multilevelGaussian(reference,
                       target,
                       params,
                       nBatch=None,
                       scales=None,
                       plist=None,
                       system="HOST"):
    if scales is None:
        scales = [0.1, 0.1]
    plist = [0.1, 0.01, 0.01]
    return False


def setupParams(angle_range, nangle, scale_range, nscale):
    params = np.asarray([
        math.radians(angle_range[0]),
        math.radians(angle_range[1]), nangle, scale_range[0], scale_range[1],
        nscale, scale_range[0], scale_range[1], nscale
    ],
                        dtype=np.float64)
    return params


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def compute_bbox(mask, quad=False):
    x_min = np.max(first_nonzero(mask, 1))
    x_max = mask.shape[1] - np.max(first_nonzero(mask[:, ::-1], 1))

    y_min = np.max(first_nonzero(mask, 0))
    y_max = mask.shape[0] - np.max(first_nonzero(mask[::-1, :], 0))

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    if quad:
        a = np.zeros(8, dtype=np.int32)
        
        a[0] = x_min
        a[1] = y_min
        
        a[2] = x_min
        a[3] = y_max
        
        a[4] = x_max
        a[5] = y_max
        
        a[6] = x_max
        a[7] = y_min
        
        return (x_min, y_min, x_max, y_max)
    return (x_min, y_min, x_max-x_min, y_max-y_min)

def compute_mask(bbox, shape, boundary=0):
    mask =  np.zeros(shape, dtype=np.uint8)
    x_min, y_min, x_len, y_len = bbox
    x_min= x_min+boundary
    y_min = y_min+boundary
    x_len = x_len - 2*boundary
    y_len = y_len - 2*boundary
    mask[y_min:y_min+y_len, x_min:x_min+x_len] = 1
    return mask

def makeResultList(gpu_output):
    result_list = []
    peak = 0.0
    for i in range(len(gpu_output)):
        peak = gpu_output[i][0]
        soln = (gpu_output[i][1], gpu_output[i][2], gpu_output[i][3], gpu_output[i][4], gpu_output[i][5])
        new_result = Result(peak, soln)
        result_list.append(new_result)
    return result_list
        
def singleSearch(reference,
                 moving,
                 params,
                 nBatch=100,
                 scale=0.5,
                 system="HOST",
                 masks=None,
                 record=False, 
                 k = 5,
                 search_type="GPC",
                 refine=True
                 ):
    if masks is not None:
        mask_r, mask_m = masks
    
    scaling = 1
    plotting = False

    #Reduce Image Resolution to accelerate search 
    m = multilevel_reduce(moving, moving.shape[0] * scale)
    r = multilevel_reduce(reference, reference.shape[0] * scale)

    if masks is not None:
        mm = multilevel_reduce(mask_m, m.shape[0])
        mr = multilevel_reduce(mask_r, r.shape[0])
        reduced_masks = (mr, mm)
    else:
        reduced_masks = None

    corr_surface = None
    start_time = time.perf_counter()

    if system is "HOST":
        print("mbox", compute_bbox(reduced_masks[1]))
        p = runSearch(r, m, params, masks=reduced_masks, record=record, top_k=k, search_type=search_type)
        corr_surface = p[1]
        p_list = p[0]
    elif system is "GPU":
        rbbox = compute_bbox(mr, quad=False)
        mbbox = compute_bbox(mm)
        #implot(mm)
        print("rbox", rbbox)
        print("mbox", mbbox)
        #raise Exception()
        p = pc.runGPUSearch(
            r, 
            rbbox,
            m, 
            mbbox, 
            params, 
            nBatch, 
            k=k, 
            search_type=search_type)

        p_list = makeResultList(p)
        
    end_time = time.perf_counter() 
    search_time = end_time - start_time
    
    if refine:
        #Compute Full Resolution Refinement
        r = reference
        rd = gradient(r)
        
        if masks is not None:
            rd = mask_r * rd
        
        for result in p_list:
            angle, sw, sh, dx, dy = result.params
            
            #r, m, loc = padding(reference, moving, scaling=scaling, dtype=np.uint8)
            m = moving
            #warp moving image
            m = warp(m, [0, 0, angle, 1, 1])
            m = warp(m, [0, 0, 0, sw, sh])
    
            #warp masks
            if masks is not None:
                mm = warp(mask_m, [0, 0, angle, 1, 1])
                mm = warp(mm, [0, 0, 0, sw, sh])
    
            md = gradient(m)
            
            if masks is not None:
                md = mm * md
    
            if search_type is "NGF":
                dx, peak, corr = NGF(   
                                        rd,
                                        md,
                                        scaling=1,
                                        label="Final Alignment",
                                        windowing=False,
                                        masks=(mask_r, mm),
                                        scales=(sw, sh),
                                        plot=False
                                    )
            elif search_type is "MeanNGF":
                    dx, peak, corr = MeanNGF(rd,
                                            md,
                                            plot=False,
                                            scaling=1,
                                            windowing=False,
                                            masks=(mask_r, mm),
                                            scales=(sw, sh)
                                            )
            elif search_type is "GPC":
                dx, peak, corr = phase_correlation(
                                                    rd,
                                                    md,
                                                    scaling=1,
                                                    label="Final Alignment",
                                                    windowing=False,
                                                    masks=(mask_r, mm),
                                                    plot=False
                                                  )
            elif search_type is "GCC":
                dx, peak, corr = cross_correlation(
                                                    rd,
                                                    md,
                                                    scaling=1,
                                                    label="Final Alignment",
                                                    windowing=False,
                                                    masks=(mask_r, mm),
                                                    plot=False,
                                                    mean=False,
                                                    scales=(sw, sh),
                                                    std=False
                                                  )
            elif search_type is "GCC_mean":
                dx, peak, corr = cross_correlation(
                                        rd,
                                        md,
                                        scaling=1,
                                        label="Final Alignment",
                                        windowing=False,
                                        masks=(mask_r, mm),
                                        plot=False,
                                        mean=True,
                                        scales=(sw, sh),
                                        std=False
                                      )
            elif search_type is "GCC_std":
                dx, peak, corr = cross_correlation(
                        rd,
                        md,
                        scaling=1,
                        label="Final Alignment",
                        windowing=False,
                        masks=(mask_r, mm),
                        plot=False,
                        mean=True,
                        scales=(1, 1),
                        std=True
                      )
            elif search_type is "CC":
                dx, peak, corr = cross_correlation(
                                                    r,
                                                    m,
                                                    scaling=1,
                                                    label="Final Alignment",
                                                    windowing=False,
                                                    masks=(mask_r, mm),
                                                    plot=False,
                                                    mean=False,
                                                    scales=(1, 1),
                                                    std=False
                                                  )
            elif search_type is "CC_mean":
                dx, peak, corr = cross_correlation(
                                    r,
                                    m,
                                    scaling=1,
                                    label="Final Alignment",
                                    windowing=False,
                                    masks=(mask_r, mm),
                                    plot=False,
                                    mean=True,
                                    std=False,
                                    scales=(1, 1)
                                  )
            elif search_type is "NCC":
                    dx, peak, corr = normed_cross_correlation(r,
                                   m, 
                                   plot=False,
                                   windowing=False,
                                   masks=(mask_r, mm)
                                )
            
            m = warp(np.abs(m), [dx[1], dx[0], 0, 1, 1])
            result_mask = warp(mm, [dx[1], dx[0], 0, 1, 1])
            result.image = m
            result.mask = result_mask
            result.params = (angle, sw, sh, dx[1], dx[0])
            result.corr = corr
    else:
        for result in p_list:
            angle, sw, sh, dx, dy = result.params
            result.image = warp(m, [0, 0, angle, 1, 1])
            result.image = warp(result.image, [0, 0, 0, sw, sh])
            result.image = warp(result.image,[dy, dx, 0, 1, 1])
            result.ref = r
        
    return p_list, corr_surface

def multilevelRoutine(reference,
                      target,
                      params,
                      nBatch=None,
                      scales=None,
                      plist=None,
                      system="HOST",
                      masks=None,
                      record=False, 
                      refine=True):

    mr = None
    mt = None

    if masks is not None:
        mr, mt = masks

    if scales is None:
        scales = [0.5]
        plist = [0.1, 0.01, 0.01]
        rate = [4, 1, 1]

    levels = len(scales)

    if nBatch is None:
        nBatch = [100] * levels

    scaling = 1
    level = 0

    angle = 0
    sw = 1
    sh = 1
    p = np.asarray([angle, sw, sh])

    for s in scales:

        t = multilevel_reduce(target, target.shape[0] * s)
        r = multilevel_reduce(reference, reference.shape[0] * s)

        r, t, loc = padding(r, t, scaling=scaling, dtype=np.uint8)

        print("Reference shape: ", r.shape)
        if masks is not None:
            tmt = multilevel_reduce(mt, t.shape[0])
            tmr = multilevel_reduce(mr, r.shape[0])
            print("Mask shape: ", mr.shape)
        #implot(r)
        #implot(t)

        start = time.time()
        if system is "HOST":
            if masks is not None:
                p = runSearch(r, t, params, masks=(tmr, tmt), record=record)
            else:
                p = runSearch(r, t, params, record=record)
            if record:
                surface = p[1]
                l = p
                p = p[0]
                lrcorr = p[2]
            else:
                l = p
                p = p[0]
                lrcorr = p[1]
        else:
            #p = pc.runSearch_GPU(r, t, params, nBatch[level], 2);
            l = p
            p = p[0]
            surface = None
            print(p)

        end = time.time()

        print("Iteration Took: ", end - start, " (s) ")

        #Adjust params
        perc = plist[level]

        angle = p[0]

        lower_angle = angle - angle * perc
        upper_angle = angle + angle * perc
        params[0] = lower_angle
        params[1] = upper_angle
        params[2] = params[2] / rate[level]
        #print("New angle range:", (lower_angle, upper_angle, params[2]))

        sw = p[1]
        lower_sw = sw - sw * perc
        upper_sw = sw + sw * perc
        params[3] = lower_sw
        params[4] = upper_sw
        params[5] = params[5] / rate[level]
        #print("New scale_w range:", (lower_sw, upper_sw, params[5]))

        sh = p[2]
        lower_sh = sh - sh * perc
        upper_sh = sh + sh * perc
        params[6] = lower_sh
        params[7] = upper_sh
        params[8] = params[8] / rate[level]
        #print("New scale_w range:", (lower_sh, upper_sh, params[8]))

        print("At level ", level, " the estimate is: ", p)
        level = level + 1

    r, t, loc = padding(reference, target, scaling=scaling, dtype=np.uint8)
    #print("2", r.shape, t.shape)
    t = warp(target, [0, 0, angle, 1, 1])
    t = warp(t, [0, 0, 0, sw, sh])

    #warp masks
    if masks is not None:
        mt = warp(mt, [0, 0, angle, 1, 1])
        mt = warp(mt, [0, 0, 0, sw, sh])

    rd, td = derivative(reference, t, method='Difference')
    #implot(rd, "rd")
    if masks is not None:
        rd = mr * rd
        td = mt * td

    #implot(rd, "rd")
    #implot(td, "td")
    if use_ngf:
        dx, peak, corr = NGF(rd,
                                             td,
                                             scaling=1,
                                             label="Final Correlation",
                                             windowing=False,
                                             mask=(mr, mt),
                                             plot=False)
    else:
        dx, peak, corr = phase_correlation(rd,
                                           td,
                                           scaling=1,
                                           label="Final Correlation",
                                           windowing=False,
                                           mask=(mr, mt),
                                           plot=False)
    print("Readjusting rescaled image by translation: ", dx)
    t = warp(np.abs(t), [dx[1], dx[0], 0, 1, 1])
    print(dx, peak)
    p = [dx[1], dx[0], angle, sw, sh]

    print("t", t.shape)
    if record:
        return t, l, surface, corr
    return t, l, corr


def post_warp(reference, img, params, scaling=1):
    dy = params[0]
    dx = params[1]
    angle = params[2]
    sw = params[3]
    sh = params[4]

    print("post warp params", (angle, sw, sh))
    #reference, img = padding(reference, img, scaling=scaling, dtype=np.uint8)
    img = warp(img, [0, 0, angle, 1, 1])
    img = warp(img, [0, 0, 0, sw, sh])
    img = warp(img, [dy, dx, 0, 1, 1])

    return img


def implot(x, title=None, line=None):

    if len(x.shape) > 2 and x.shape[2] == 1:
        x = x.reshape(x.shape[0], x.shape[1])

    fig = plt.figure()
    plt.imshow(np.abs(x), cmap='gray')
    if title is not None:
        plt.title(title)
    if line:
        plt.axhline(y=line, color='red', alpha=0.5)
    plt.show()


def added_plot(ref, mov, title="Registered Overlay", alpha=0.5):
    ref = np.abs(ref)
    mov = np.abs(mov)
    ref, mov, loc = padding(ref, mov, dtype=np.uint8)
    added_image = cv2.addWeighted(ref, alpha, mov, 1 - alpha, 0)
    implot(added_image, title)


def create_checker(image, width=None, height=None):
    out = np.zeros(image.shape)
    if width is None:
        width = image.shape[1] / 5
    if height is None:
        height = image.shape[0] / 5

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if bool(int(i / height) % 2 == 0) ^ bool(int(j / width) % 2 == 0):
                out[i, j] = 1

    return out


def rasterToImage(raster):
    if raster is not None:
        arr = None
        for i in range(0, (raster.RasterCount)):
            band = raster.GetRasterBand(i + 1)
            print(type(band.ReadAsArray()))
            print(band.ReadAsArray().dtype)

            if arr is None:
                arr = np.asarray(band.ReadAsArray(),
                                 dtype=np.float32) / raster.RasterCount
            else:
                arr += np.asarray(band.ReadAsArray(),
                                  dtype=np.float32) / raster.RasterCount

        return cv2.convertScaleAbs(arr, alpha=255 / arr.max())
    else:
        raise Exception(
            "Attempting to access empty or non-existant raster file.")


def linearRemap(I, k):
    try:
        height, width = I.shape
    except ValueError:
        height = I.shape
        width = None

    anchors = np.sort(np.random.rand(k)) * 255
    anchors2 = np.append(anchors, 255)
    d = 255 - np.diff(anchors2)
    remaps = np.random.rand(k) * d

    I = np.ravel(I)
    idx = np.searchsorted(anchors, I)
    J = remaps[idx - 1] + (I - anchors[idx - 1])
    if width is not None:
        return J.reshape(height, width)
    else:
        return J


def gaussianRemap(I, k1, k2):
    psigma = 20**2
    ssigma = 3 * 100**2
    means_p = np.random.rand(k1) * 255
    means_s = np.random.rand(k2, 2)

    M, N = I.shape
    means_s[:, 0] = means_s[:, 0] * M
    means_s[:, 1] = means_s[:, 1] * N

    x = np.linspace(0, N, N, endpoint=False)
    y = np.linspace(0, M, M, endpoint=False)
    xv, yv = np.meshgrid(x, y)

    mask_p = np.ones((M, N))
    mask_s = np.ones((M, N))

    if k1 > 0:
        mask_p = 1 / k1 * np.ones((M, N))
        for m in means_p[:]:
            #m = np.array([40, 40])
            #implot(1/k * np.exp( -((xv - m[1])**2 + (yv-m[0])**2)/sigma ))
            #mask += 1/k * np.exp(-((xv - m[1])**2 + (yv-m[0])**2)/sigma )
            mask_p += 1 / k1 * np.exp(-((I - m)**2) / psigma)

    if k2 > 0:
        mask_s = 1 / k2 * np.ones((M, N))
        for m in means_s[:]:
            mask_s += 1 / k2 * np.exp(-((xv - m[1])**2 +
                                        (yv - m[0])**2) / ssigma)

    return I * mask_p * mask_s

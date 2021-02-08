# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv2

import scipy.ndimage.interpolation as ndii
from scipy import interpolate

import matplotlib.pyplot as plt

import matplotlib.style
import matplotlib as mpl
import matplotlib.cm as cm
import math
import time

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 400
mpl.rcParams['savefig.dpi'] = 450
mpl.rcParams['hatch.linewidth'] = 1.0
mpl.rcParams['font.size'] = 16
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

from image_helper import *

plotting = False
save = False

#Load image and flatten

folder = 'residential_crop/c1/'
#folder = 'rome_crop/intensity/'
folder2 = 'test_crop'



#reference = 255*np.ones((300, 300), dtype=np.uint8) #cv2.imread('rome_image.png')
#moving = 255*np.ones((100, 100), dtype=np.uint8) #cv2.imread(folder+'crop_'+str(0)+'.png')


#reference = cv2.imread('large.png.png')
#reference = cv2.imread('house1.png')
#moving = cv2.imread('house3.png')
#reference = cv2.imread('rome_image.png')
#moving = cv2.imread('small_crop.png')
moving = cv2.imread(folder+'crop_14.png')
#moving = cv2.imread('rome_image_crop_3.png')
#moving = cv2.imread()
#moving = reference
#reference = cv2.imread('rome_image.png')
#reference = cv2.imread('rome_intensity_rescaled_2.png')
reference = cv2.imread('Opt_residential.png')
#reference = moving
#reference = cv2.imread('houston_optical.png')
#moving = cv2.imread('houston_intensity.png')
#moving = reference.copy()

reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
moving = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)

#reference = moving
#moving = np.load("rome_level.npy")
#reference = multilevel_reduce(reference, 256)
#moving = multilevel_reduce(moving, reference.shape[0])

box = compute_bbox(moving)
moving = box_to_crop(box, moving, reverse=True)

print("Reference Shape is ", reference.shape)
print("Moving Shape is ", moving.shape)

reference, moving, mr, mm, loc = padding(reference,
                                         moving,
                                         scaling=2,
                                         dtype=np.float32,
                                         mask=True,
                                         window=False)

box = compute_bbox(moving)
print(box)

#box = [box[0]+128, box[1]+128, box[2]-256, box[3]-256]
mm = compute_mask(box, moving.shape, boundary=5)
print(box)
implot(moving*mm, "New Mask")
low = 0.6
up = 1.4
scales = np.linspace(low, up, 11)

a = 0
sx = 1/1.4#/scales[5]
sy = 1/0.68#/scales[5]
dx = 0
dy = 0

moving = warp(moving,[dx, dy, math.radians(a), sx, sy])
mm = warp(mm, [dx, dy, math.radians(a), sx, sy])

#reference = warp(reference, [dx, dy, math.radians(a), sx, sy])
#mr = warp(mr, [dx, dy, math.radians(a), sx, sy])


pt = np.asarray([-math.radians(a), 1 / sx, 1 / sy])
angle_range = [0, 0]
#angle_range = [0, 0]
#scale_range = [0.5, 1 / 0.5]
#scale_range = [0.8, 1.2]
#angle_range = [0, 0]
#scale_range = [0.5, 1.5]
#scale_range = [0.5, 4]
#scale_range = [low, up]
scale_range = [0.6, 1.4]
use_ngf = False
nangle = 1
nscale = 11
angle_plot = False
scale_plot = False

if nangle > 1:
    angle_plot = True
if nscale > 1:
    scale_plot = True

nbatch = [100, 100, 100, 100]
scales = None

l = time.perf_counter()

params = np.asarray([
    math.radians(angle_range[0]),
    math.radians(angle_range[1]),
    nangle,
    scale_range[0],
    scale_range[1],
    nscale,
    scale_range[0],
    scale_range[1],
    nscale
    ],
    dtype=np.float64)


refine_flag = True
#implot(moving*mm, "moving")
#implot(reference*mr, "ref")

p_list, surface = singleSearch(
    reference,
    moving, 
    params,
    scale = 0.5,
    system="HOST",
    masks=(mr, mm),
    record=True,
    k=5,
    search_type="MeanNGF",
    refine=refine_flag,
    nBatch=17
)

l = time.perf_counter() - l

print("Total Time: ", l)
print(p_list)

if refine_flag is not False:
    added_plot(reference, p_list[0].image)
else:
    added_plot(p_list[0].ref, p_list[0].image)
    fig = plt.figure()
    plt.imshow(np.fft.fftshift(np.abs(p_list[0].corr)))
#implot(added_image, "Registered Overlay")

if surface is not None:
    Dy, Dx, Dz = np.unravel_index(surface.argmax(), surface.shape)
    print("Surface Max at : ", Dy, Dx, Dz)
    surface = surface / surface[Dy, Dx, Dz]
    if angle_plot:
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.linspace(angle_range[0], angle_range[1], nangle),
                surface[:, Dx, Dz])
        ax.xaxis.set_major_locator(plt.MultipleLocator(60))
        #ax.xaxis.set_minor_locator(plt.MultipleLocator(5))
        plt.xlabel('Angle (Degrees)')
        plt.title('Similiarity')
        plt.show()
    if scale_plot:
        fig, ax = plt.subplots(1, 1)
        extent = scale_range + scale_range
        x = np.linspace(scale_range[0], scale_range[1], nscale+1)
        ax.pcolor(x, x, surface[Dy, :, :], cmap=cm.gray)
        #ax.contour(x, x, surface[Dy, :, :], origin='lower')
        #plt.xlabel('Scale (Height)')
        #plt.ylabel('Scale (Width)')
        plt.show()

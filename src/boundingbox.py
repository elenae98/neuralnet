import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter

from skimage import exposure

import torch.nn as nn
import torchvision

from PIL import Image
import glob as glob
import matplotlib.pyplot as plt
from numpy import genfromtxt
import scipy.misc



'''import image'''
imgArray = genfromtxt('../data/(기준)22-Feb-2018 15-32-02 Surface Phase6 TypeBoth Camera1.map', delimiter=',')

plt.imsave(imgArray, '../data/exImage.png')

# flatImg = np.ndarray.flatten(imgArray[~np.isnan(imgArray)])
# # print(flatImg.shape)
# # hist = np.histogram(flatImg)
# plt.hist(flatImg)
# plt.show()


'''clipping pixel values to a smaller range'''
v_min, v_max = np.percentile(imgArray[~np.isnan(imgArray)], (0.75, 98)) #clip
# '''amplify contrast'''
# # imgArray = exposure.rescale_intensity(imgArray, in_range=(v_min, v_max)) #contrast
# # plt.subplot(221) #plot to show on figure later
# # plt.imshow(imgArray)
imgArray[imgArray<v_min] = v_min
imgArray[imgArray>v_max] = v_max




'''normalize array values to fit color map 0-255'''
min_val = np.min(imgArray[~np.isnan(imgArray)])
max_val = np.max(imgArray[~np.isnan(imgArray)])
print(min_val)
print(max_val)

imgArray -= min_val
imgArray *= 255 / (max_val - min_val)

min_val = np.min(imgArray[~np.isnan(imgArray)])
max_val = np.max(imgArray[~np.isnan(imgArray)])
print(min_val)
print(max_val)

#hist
flatImg = np.ndarray.flatten(imgArray[~np.isnan(imgArray)])
plt.hist(flatImg)
plt.show() #mode is between 130 and 150


'''makes mode 0 and omits'''
# modeArray = imgArray[imgArray>130]
# modeArray = modeArray[modeArray<150]

max_arr = imgArray
max_arr[np.isnan(imgArray)] = 0
max_arr[max_arr < 150] = 0

min_arr = imgArray
min_arr[np.isnan(imgArray)] = 0
min_arr[imgArray > 130] = 0
outlier_arr = max_arr + min_arr

outlier_arr[outlier_arr == 0] = np.nan

plt.subplot(221)
var1 = plt.imshow(outlier_arr, cmap="inferno")
plt.colorbar(var1)
plt.show()
#
# '''median filter to blur small details out'''
# imgArray = median_filter(imgArray, 3)
# plt.subplot(222)
# plt.imshow(imgArray[(imgArray<130 and imgArray>150)], cmap="inferno")
#
# '''difference of gaussian blur filters for edge detection'''
# gauss1 = gaussian_filter(imgArray, sigma=3)
# gauss2 = gaussian_filter(imgArray, sigma=1)
# subt = gauss2-gauss1
# plt.subplot(223)
# plt.imshow(subt[(imgArray<130 and imgArray>150)], cmap="inferno")
# plt.show()
#
#
#
# # make a filter the size of the image
#
#
# # find gradients with different sigmas and subtract the filters from ea other

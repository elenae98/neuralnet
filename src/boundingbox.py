import numpy as np
from scipy.ndimage import median_filter
import cv2
import matplotlib.pyplot as plt
from numpy import genfromtxt

'''colomap'''
colormap = 'gist_ncar'

'''import image'''
imgArray = genfromtxt('../data/(기준)22-Feb-2018 15-32-02 Surface Phase6 TypeBoth Camera1.map', delimiter=',')

'''clipping pixel values to a smaller range'''
v_min, v_max = np.percentile(imgArray[~np.isnan(imgArray)], (0.75, 98)) #clip
imgArray[imgArray<v_min] = v_min
imgArray[imgArray>v_max] = v_max

'''normalize array values to fit color map 0-255'''
min_val = np.min(imgArray[~np.isnan(imgArray)])
max_val = np.max(imgArray[~np.isnan(imgArray)])
imgArray -= min_val
imgArray *= 255 / (max_val - min_val)

'''show original picture'''
plt.subplot(221)
var1 = plt.imshow(imgArray, cmap=colormap)
plt.colorbar(var1)

'''median filter to blur small details out'''
imgArray = median_filter(imgArray, 3)
plt.subplot(222)
plt.imshow(imgArray, cmap=colormap)
plt.show()

# '''difference of gaussian blur filters for edge detection'''
# gauss1 = gaussian_filter(imgArray, sigma=3)
# gauss2 = gaussian_filter(imgArray, sigma=1)
# subt = gauss2-gauss1
# plt.subplot(223)
# plt.imshow(subt, cmap=colormap)
# plt.show()



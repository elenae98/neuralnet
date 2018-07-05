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

# Gaussian image detector

# count = 1
'''import image'''

# imageArrays = []

# for pic in glob.glob('../cats/CAT_00/*.jpg'):
#    img = Image.open(pic)
#     imgArray = np.array(img)

imgArray = genfromtxt('./data/', delimiter=',')

plt.figure(count)

v_min, v_max = np.percentile(imgArray, (30, 70))
imgArray = exposure.rescale_intensity(imgArray, in_range=(v_min, v_max))
plt.subplot(221)
plt.imshow(imgArray)

imgArray = median_filter(imgArray, 7)
plt.subplot(222)
plt.imshow(imgArray)

result = gaussian_filter(imgArray, sigma=7)
plt.imshow(result)
# plt.show()

result2 = gaussian_filter(imgArray, sigma=3)
subt = result - result2

plt.subplot(223)
plt.imshow(subt)
plt.show()

'''
round2a = gaussian_filter(subt, sigma = 7)
round2b = gaussian_filter(subt, sigma = 2)
subt2 = round2a - round2b
plot2 = plt

plot2.imshow(subt2)
plot2.show()

'''
Image.Image.close(img)
imageArrays.append(imgArray)
count += 1
if count == 5:
    break

# make a filter the size of the image


# find gradients with different sigmas and subtract the filters from ea other

import numpy as np
from scipy.ndimage import median_filter
import cv2
import matplotlib.pyplot as plt
from numpy import genfromtxt

'''selected colomap'''
colormap = 'gist_ncar'

'''import image'''
imgArray = genfromtxt('../data/(기준)22-Feb-2018 15-32-02 Surface Phase6 TypeBoth Camera1.map', delimiter=',')

'''clipping pixel values to a smaller range'''
v_min, v_max = np.percentile(imgArray[~np.isnan(imgArray)], (0.8, 97))  # clip
imgArray[imgArray < v_min] = v_min
imgArray[imgArray > v_max] = v_max

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
imgOrig = imgArray
imgArray = median_filter(imgArray, 4)
plt.subplot(222)
plt.imshow(imgArray, cmap=colormap)
plt.show()

'''bounding box generator'''
'''take avg of sliced col and row'''
rows, cols = np.shape(imgArray)  # 2048, 2592
# rows
row_avgs = []
for i in range(rows):
    rowTemp = imgArray[i, :]
    if len(rowTemp[~np.isnan(rowTemp)]) == 0:
        row_avgs.append(np.nan)
    else:
        row_avgs.append(np.mean(rowTemp[~np.isnan(rowTemp)]))
# cols
col_avgs = []
for i in range(cols):
    colTemp = imgArray[:, i]
    if len(colTemp[~np.isnan(colTemp)]) == 0:
        col_avgs.append(np.nan)
    else:
        col_avgs.append(np.mean(colTemp[~np.isnan(colTemp)]))

'''make picture binary'''


'''max/min rows & cols -> bumps on surface'''
# max_col = col_avgs.index(np.nanmax(col_avgs))
# max_row = row_avgs.index(np.nanmax(row_avgs))
# min_col = col_avgs.index(np.nanmin(col_avgs))
# min_row = row_avgs.index(np.nanmin(row_avgs))
#
# print(np.nanmax(col_avgs))
# print(max_col)
# print('bump avgs: top edge of top right circle ', row_avgs[950])
# print('bump avgs: red left col of top right circle ', col_avgs[1900])

#
# # tempImg = imgArray
# '''col = row interchangeable use in this fcn// find val for uppermax percentile, etc. & map'''
# def percentile_finder(avg, uppermax, lowermax, col=True, uppermin=0, lowermin=0):
#     umax = np.nanpercentile(avg, uppermax)
#     lmax = np.nanpercentile(avg, lowermax)
#     print(uppermax, 'th: ', umax, ' ', lowermax, ' th: ', lmax)
#
#     umin = np.nanpercentile(avg, uppermin)
#     lmin = np.nanpercentile(avg, lowermin)
#     print(uppermin, 'th: ', umin, ' ', lowermin, ' th: ', lmin)
#
#     i_lmax = np.argwhere(avg > lmax)
#     i_umax = np.argwhere(avg < umax)
#     maxThresh = list(filter(lambda x: x in i_lmax, i_umax))
#
#     i_lmin = np.argwhere(avg > lmin)
#     i_umin = np.argwhere(avg < umin)
#     minThresh = list(filter(lambda x: x in i_lmin, i_umin))
#
#     if col: #make vals in maxThresh white, make v in minThresh black
#         for c in maxThresh:
#             for r in range(rows):
#                 tempImg[r][c] = 255
#         for c in minThresh:
#             for r in range(rows):
#                 tempImg[r][c] = 0
#     else:
#         for c in range(cols):
#             for r in maxThresh:
#                 tempImg[r][c] = 255
#         for c in range(cols):
#             for r in minThresh:
#                 tempImg[r][c] = 0
#     return tempImg


# tempImg = percentile_finder(col_avgs, 78, 55)
# plt.subplot(223)
# plt.imshow(tempImg, cmap=colormap)
# plt.show()

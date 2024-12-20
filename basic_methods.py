import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def nearestNeighbors(image, mask, neighborhood_size = 5):
    result = image.copy()
    imgW, imgH = image.shape

    bad_pixels = np.where(mask == 0)

    for i, j in zip(*bad_pixels):
        z = neighborhood_size

        #neighborhood box
        l = max(0, i-z)
        r = min(imgW, i+z+1)
        t = max(0, j-z)
        b = min(imgH, j+z+1)
        neighborhood = result[l:r, t:b]

        nearby_mean = np.mean(neighborhood[mask[l:r, t:b] == 255])

        result[i, j] = nearby_mean

    return result



### GAUSSIAN

def gaussianInpaint(img, mask, sigma=15, radius = 20):
    result = img.copy()
    result[mask==0] = np.mean(img[mask>0])

    blurred = ndimage.gaussian_filter(result, sigma=sigma, radius=20)
    result[mask == 0] = blurred[mask == 0]

    return result 
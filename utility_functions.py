
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as tfs
import torchvision.transforms.functional as tF
import random
from PIL import Image


def temp(img):
    return 5

def thinXMask(img):
    thinMask = np.zeros(img.shape)

    iW, iH = img.shape
    x = 10

    for i in range(20, iW-20):
        x += 1
        thinMask[i,x:x+2] = 1
        thinMask[-i, x: x+2] = 1

    return thinMask

def thickXMask(img):
    thickMask = np.zeros(img.shape)

    iW, iH = img.shape
    x = 10

    for i in range(20, iW-20):
        x += 1
        thickMask[i,x:x+10] = 1
        thickMask[-i, x: x+10] = 1

    return thickMask


def showResults(orig, mask, result, color=False):
    plt.figure(figsize=(20, 8))
    plt.subplot(131)
    plt.imshow(orig, cmap='gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(result, cmap='gray', vmin=0, vmax=255)
    plt.title('Inpainted Result')
    plt.axis('off')

    plt.tight_layout()




class RandomHFlip(object):
    def __call__(self, pair):
        img, target = pair
        num = random.randint(0, 1)
        if num == 1:
            return tF.hflip(img), tF.hflip(target)
        else:
            return img, target

class RandomVFlip(object):
    def __call__(self, pair):
        img, target = pair
        num = random.randint(0, 1)
        if num == 1:
            return tF.vflip(img), tF.vflip(target)
        else:
            return img, target  


class PairColorJitter(object):

    def __call__(self, pair):
        img, target = pair

        brightness = random.uniform(0.2, 1.8)
        contrast = random.uniform(0.2, 1.8)
        saturation = random.uniform(0.2, 5.8)
        hue = random.uniform(-0.5, 0.5)

        img = tF.adjust_brightness(img, brightness)
        img = tF.adjust_contrast(img, contrast)
        img = tF.adjust_saturation(img, saturation)
        img = tF.adjust_hue(img, hue)

        target = tF.adjust_brightness(target, brightness)
        target = tF.adjust_contrast(target, contrast)
        target = tF.adjust_saturation(target, saturation)
        target = tF.adjust_hue(target, hue)

        return img, target


# NOT CURRENTLY USED
class RandomResizeCrop(object):
    
    def __init__(self, size, min_scale, max_scale):
        self.size = size
        self.min_scale = min_scale
        self.max_scale = max_scale
        

    def __call__(self, pair):
        img, target = pair
        new_size = (int(img.size[0] * random.uniform(self.min_scale, self.max_scale)),
                                     int(img.size[1] * random.uniform(self.min_scale, self.max_scale)))
        
        scaled_img = tF.resize(img, new_size)
        scaled_target = tF.resize(target, new_size)
        final_size = min(self.size, scaled_img.size[0], scaled_img.size[1])
        crop_width = random.randint(0, scaled_img.size[1] - final_size)
        crop_height = random.randint(0, scaled_img.size[0] - final_size)
        
        return (tF.resized_crop(scaled_img, crop_width, crop_height, final_size, final_size, self.size),
                tF.resized_crop(scaled_target, crop_width, crop_height, final_size, final_size, self.size),)
      
        
class MakePair(object):
  def __call__(self, img):
    target = img.copy()

    return  Image.fromarray(img), Image.fromarray(target) 

class RectanglePatch(object):
  def __init__(self, patch_size):
      self.patch_size = patch_size

  def __call__(self, pair):

    img, target = pair

    #orig = img.copy()
    new_img = np.array(img)

    imgW, imgH, _ = new_img.shape
    patchW, patchH = self.patch_size

    l = (imgW // 2) - (patchW // 2)
    r = l + patchW
    t = (imgH // 2) - (patchH // 2)
    b = t + patchH

    new_img[l:r, t:b] = 255

    res = Image.fromarray(new_img) 

    return (res, target)
  


class CrossPatch(object):
  def __init__(self, patch_size):
    self.patch_size = patch_size

  def __call__(self, pair):
    img, target = pair

    new_img = np.array(img)

    iW, iH,_ = new_img.shape
    x = 10

    for i in range(20, iW-20):
        x += 1
        new_img[i,x:x + self.patch_size] = 255
        new_img[-i, x: x + self.patch_size] = 255

    res = Image.fromarray(new_img) 

    return res, target

  
class PairToTensors(object):
  def __call__(self, pair):
    img, target = pair
    return tfs.ToTensor()(img), tfs.ToTensor()(target)



class PairNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, pair):
        tensor, target = pair
        return (tF.normalize(tensor, self.mean, self.std), tF.normalize(target, self.mean, self.std))


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
    thinMask = np.full(img.shape, fill_value=255)

    iW, iH = img.shape
    x = 10

    for i in range(20, iW-20):
        x += 1
        thinMask[i,x:x+2] = 0
        thinMask[-i, x: x+2] = 0

    return thinMask

def thickXMask(img):
    thickMask = np.full(img.shape, fill_value=255)

    iW, iH = img.shape
    x = 10

    for i in range(20, iW-20):
        x += 1
        thickMask[i,x:x+10] = 0
        thickMask[-i, x: x+10] = 0

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
        img, mask = pair
        num = random.randint(0, 1)
        if num == 1:
            return tF.hflip(img), tF.hflip(mask)
        else:
            return img, mask

class RandomVFlip(object):
    def __call__(self, pair):
        img, mask = pair
        num = random.randint(0, 1)
        if num == 1:
            return tF.vflip(img), tF.vflip(mask)
        else:
            return img, mask  


class PairColorJitter(object):

    def __call__(self, pair):
        img, mask = pair

        brightness = random.uniform(0.2, 1.8)
        contrast = random.uniform(0.2, 1.8)
        saturation = random.uniform(0.2, 5.8)
        hue = random.uniform(-0.5, 0.5)

        img = tF.adjust_brightness(img, brightness)
        img = tF.adjust_contrast(img, contrast)
        img = tF.adjust_saturation(img, saturation)
        img = tF.adjust_hue(img, hue)

        return img, mask


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


class CrossMask(object):
  def __init__(self, patch_size):
    self.patch_size = patch_size

  def __call__(self, img):
    mask = np.full_like(img, fill_value=255)

    iW, iH,_ = mask.shape
    x = 10

    for i in range(20, iW-20):
        x += 1
        mask[i,x:x + self.patch_size] = 0
        mask[-i, x: x + self.patch_size] = 0

    return img, mask

  
class PairToTensors(object):
  def __call__(self, pair):
    img, mask = pair
    return tfs.ToTensor()(img), tfs.ToTensor()(mask)



class PairNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, pair):
        tensor, mask = pair
        return (tF.normalize(tensor, self.mean, self.std), mask)
    
class PairAddWhiteNoise(object):

    def __init__(self, std = 0.01):
        self.std = std
        pass
        
    def __call__(self, pair):
        tensor, mask = pair
        noise = torch.rand_like(tensor, dtype=torch.float) * self.std
        
        return (torch.clamp(tensor + noise, 0.0, 1.0), mask)
    

class RectangleMask(object):
  def __init__(self, patch_size, patch_loc):
      self.patch_size = patch_size
      self.patch_loc = patch_loc

  def __call__(self, img):
    return (img, get_rectangle_mask(img, self.patch_loc, self.patch_size))

def get_rectangle_mask(img, patch_loc, patch_size): # get into util func.py
    retval = np.full_like(img, fill_value=255)
    x, y = patch_loc
    sx, sy = patch_size
    retval[x: x+sx, y:y+sy, :] = 0 
    return retval

def torch_image_to_numpy(img):
    return img.permute(1, 2, 0).cpu().detach().numpy() 
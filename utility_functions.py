
import matplotlib.pyplot as plt
import numpy as np



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


import matplotlib.pyplot as plt

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
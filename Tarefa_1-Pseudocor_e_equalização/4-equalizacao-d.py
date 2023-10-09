import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_images_and_histograms(images, titles):
    fig, ax = plt.subplots(len(images), 2, figsize=(15, 10))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        ax[i, 0].imshow(img, cmap="gray")
        ax[i, 0].axis('off')
        ax[i, 0].set_title(title)
        
        ax[i, 1].hist(img.ravel(), bins=32, range=(0.0, 256.0), ec='k')
        ax[i, 1].set_xlabel("Intensidade de Pixel")
        ax[i, 1].set_ylabel("Contagem de Pixels")
        
    plt.savefig('equalizacao_matching.png')
    plt.show()

def matching_histogram(src, dst):
    src_values, src_unique_indices, src_counts = np.unique(src.ravel(), return_counts=True, return_inverse=True)
    dst_values, dst_counts = np.unique(dst.ravel(), return_counts=True)

    src_cdf = np.cumsum(src_counts) / len(src.ravel())
    dst_cdf = np.cumsum(dst_counts) / len(dst.ravel())

    matched_values = np.interp(src_cdf, dst_cdf, dst_values)
    
    return matched_values[src_unique_indices].reshape(src.shape)

if __name__ == "__main__":
    src = cv2.imread('Imgs_Originais/fourier_1.jpg', cv2.IMREAD_GRAYSCALE)
    dst = cv2.imread('Imgs_Originais/fourier_2.jpg', cv2.IMREAD_GRAYSCALE)

    matched_histogram = matching_histogram(src, dst)
    plot_images_and_histograms([src, dst, matched_histogram], ["Fonte", "Destino", "Matched"])

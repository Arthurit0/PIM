import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_histogram_and_cdf(image, cdf_normalized, title):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.hist(image.flatten(), bins=256, range=[0,256], color='r')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Pixels')
    ax.set_xlim(0,255)
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(cdf_normalized, color='b')
    ax2.set_ylabel('cdf')
    ax2.set_ylim(0,1)
    plt.show()

def plot_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()

def process_image(image_path, title_prefix):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Histograma original
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()

    plot_histogram_and_cdf(image, cdf_normalized, f'{title_prefix} - Histograma e CDF Original')
    plot_image(image, f'{title_prefix} - Imagem Original')

    equ = cv2.equalizeHist(image)

    # Equalização Global
    hist, bins = np.histogram(equ.flatten(), bins=256, range=[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()

    plot_histogram_and_cdf(equ, cdf_normalized, f'{title_prefix} - Histograma e CDF Equalizada Global')
    plot_image(equ, f'{title_prefix} - Equalização Global')

    # Equalização local (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    clahef = clahe.apply(image)

    hist, bins = np.histogram(clahef.flatten(), bins=256, range=[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()

    plot_histogram_and_cdf(clahef, cdf_normalized, f'{title_prefix} - Histograma e CDF Equalizada Local')
    plot_image(clahef, f'{title_prefix} - Equalização Local (CLAHE)')

if __name__ == "__main__":
    process_image('Imgs_Originais/xadrez_lowCont.png', 'Xadrez Baixo Contraste')
    process_image('Imgs_Originais/XADREZ.png', 'Xadrez Alto Contraste')

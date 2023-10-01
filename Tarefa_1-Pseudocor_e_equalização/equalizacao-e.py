import cv2
import numpy as np
from skimage import color
import matplotlib.pyplot as plt

def equalize_histogram(image):
    return cv2.equalizeHist(image)

def plot_histogram(image, title):
    plt.figure()
    plt.title(title)
    for i, col in enumerate(['r', 'g', 'b']):
        hist, bins = np.histogram(image[..., i].ravel(), 256, [0, 256])
        plt.plot(hist, color=col)
    plt.show()

def main():
    # Carregar as imagens
    img_outono = cv2.imread('Imgs_Originais/outono_LC.png')
    img_predios = cv2.imread('Imgs_Originais/predios.jpeg')

    # Converter para RGB (OpenCV carrega em BGR)
    img_outono = cv2.cvtColor(img_outono, cv2.COLOR_BGR2RGB)
    img_predios = cv2.cvtColor(img_predios, cv2.COLOR_BGR2RGB)

    # 1. Equalização de contraste diretamente sobre os canais RGB
    img_outono_rgb_eq = np.zeros_like(img_outono)
    img_predios_rgb_eq = np.zeros_like(img_predios)

    for i in range(3):
        img_outono_rgb_eq[..., i] = equalize_histogram(img_outono[..., i])
        img_predios_rgb_eq[..., i] = equalize_histogram(img_predios[..., i])

    # 2. Equalização usando o sistema YIQ
    img_outono_yiq = color.rgb2yiq(img_outono)
    img_predios_yiq = color.rgb2yiq(img_predios)

    img_outono_yiq[..., 0] = equalize_histogram(img_outono_yiq[..., 0])
    img_predios_yiq[..., 0] = equalize_histogram(img_predios_yiq[..., 0])

    img_outono_yiq_eq = color.yiq2rgb(img_outono_yiq)
    img_predios_yiq_eq = color.yiq2rgb(img_predios_yiq)

    # Exibir histogramas
    plot_histogram(img_outono, 'Histograma RGB Outono Original')
    plot_histogram(img_outono_rgb_eq, 'Histograma RGB Outono Equalizado')
    plot_histogram(img_outono_yiq_eq, 'Histograma YIQ Outono Equalizado')

    plot_histogram(img_predios, 'Histograma RGB Predios Original')
    plot_histogram(img_predios_rgb_eq, 'Histograma RGB Predios Equalizado')
    plot_histogram(img_predios_yiq_eq, 'Histograma YIQ Predios Equalizado')

if __name__ == "__main__":
    main()

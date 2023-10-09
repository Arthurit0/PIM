import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import color

def plot_histogram(image, title, channel_labels=['r', 'g', 'b']):
    fig, ax = plt.subplots()
    for i, col in enumerate(channel_labels):
        hist, bins = np.histogram(
            image[..., i], bins=256, range=[0, 256]
        )
        ax.plot(hist, color=col)
    ax.set_title(title)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Pixels')
    plt.show()

def main():
    img_outono = cv2.imread('Imgs_Originais/outono_LC.png')[..., ::-1]
    img_predios = cv2.imread('Imgs_Originais/predios.jpeg')[..., ::-1]

    img_outono_yiq = color.rgb2yiq(img_outono / 255.0)
    img_predios_yiq = color.rgb2yiq(img_predios / 255.0)

    # Realizando a equalização global no canal Y (luminância)
    img_outono_yiq[..., 0] = cv2.equalizeHist((img_outono_yiq[..., 0] * 255).astype(np.uint8)) / 255.0
    img_predios_yiq[..., 0] = cv2.equalizeHist((img_predios_yiq[..., 0] * 255).astype(np.uint8)) / 255.0

    # Convertendo de volta para RGB
    img_outono_yiq_eq = color.yiq2rgb(img_outono_yiq)
    img_predios_yiq_eq = color.yiq2rgb(img_predios_yiq)

    img_outono_yiq_eq = np.clip(img_outono_yiq_eq * 255, 0, 255).astype(np.uint8)
    img_predios_yiq_eq = np.clip(img_predios_yiq_eq * 255, 0, 255).astype(np.uint8)

    # Exibir imagens e histogramas
    plt.imshow(img_outono)
    plt.title('Outono Original')
    plt.axis('off')
    plt.savefig('outono_original.png')
    plt.show()

    plot_histogram(img_outono, 'Histograma Outono Original')

    plt.imshow(img_outono_yiq_eq)
    plt.title('Outono YIQ Equalizado')
    plt.axis('off')
    plt.savefig('outono_yiq_equalizado.png')
    plt.show()

    plot_histogram(img_outono_yiq_eq, 'Histograma Outono YIQ Equalizado')

    plt.imshow(img_predios)
    plt.title('Prédios Original')
    plt.axis('off')
    plt.savefig('predios_original.png')
    plt.show()

    plot_histogram(img_predios, 'Histograma Prédios Original')

    plt.imshow(img_predios_yiq_eq)
    plt.title('Prédios YIQ Equalizado')
    plt.axis('off')
    plt.savefig('predios_yiq_equalizado.png')
    plt.show()

    plot_histogram(img_predios_yiq_eq, 'Histograma Prédios YIQ Equalizado')

if __name__ == '__main__':
    main()

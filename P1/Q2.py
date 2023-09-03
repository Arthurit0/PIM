import cv2
import numpy as np


def bayer_to_rgb(bayer_img):
    h, w = bayer_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)

    # bayer é bayer_img com a borda
    bayer = np.pad(bayer_img, ((1, 1), (1, 1)), mode='constant')

    with open("saida.txt", "w") as f:
        for i in range(1, h + 1):
            f.write(f"Linha {i}: ")
            for j in range(1, w + 1):
                if (i + j) % 2 == 0:
                    # Pixel verde
                    green = bayer[i, j]
                    red = (bayer[i-1, j] + bayer[i+1, j]) // 2
                    blue = (bayer[i, j-1] + bayer[i, j+1]) // 2
                    f.write("G")
                elif i % 2 == 0:
                    # Pixel azul
                    blue = bayer[i, j]
                    green = (bayer[i-1, j] + bayer[i+1, j] +
                             bayer[i, j-1] + bayer[i, j+1]) // 4
                    red = (bayer[i-1, j-1] + bayer[i+1, j-1] +
                           bayer[i-1, j+1] + bayer[i+1, j+1]) // 4
                    f.write("B")
                else:
                    # Pixel vermelho
                    red = bayer[i, j]
                    green = (bayer[i-1, j] + bayer[i+1, j] +
                             bayer[i, j-1] + bayer[i, j+1]) // 4
                    blue = (bayer[i-1, j-1] + bayer[i+1, j-1] +
                            bayer[i-1, j+1] + bayer[i+1, j+1]) // 4
                    f.write("R")

                rgb_img[i-1, j-1] = [blue, green, red]
            f.write("\n")
    return rgb_img


bayer_img = cv2.imread('Imagens/Lighthouse_bayerBG8.png', cv2.IMREAD_GRAYSCALE)

rgb_img = bayer_to_rgb(bayer_img)

cv2.imshow('Imagem de Entrada', bayer_img)
cv2.imshow('Imagem de Saida (com bayer)', rgb_img)
cv2.imwrite('LighthouseQ2.png', rgb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np


def bayer_para_rgb(bayer_img):
    h, w, c = bayer_img.shape

    rgb_img = bayer_img.copy()

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if not i % 2:
                if j % 2:
                    rgb_img[i, j, 2] = (
                        bayer_img[i - 1, j, 2] / 2.0 + bayer_img[i + 1, j, 2] / 2.0)
                    rgb_img[i, j, 0] = (
                        bayer_img[i, j - 1, 0] / 2.0 + bayer_img[i, j + 1, 0] / 2.0)
                else:
                    rgb_img[i, j, 2] = (bayer_img[i - 1, j - 1, 2] / 4.0 + bayer_img[i - 1, j + 1, 2] /
                                        4.0 + bayer_img[i + 1, j - 1, 2] / 4.0 + bayer_img[i + 1, j + 1, 2] / 4.0)
                    rgb_img[i, j, 1] = (bayer_img[i - 1, j, 1] / 4.0 + bayer_img[i, j + 1, 1] /
                                        4.0 + bayer_img[i, j - 1, 1] / 4.0 + bayer_img[i + 1, j, 1] / 4.0)
            else:
                if not j % 2:
                    rgb_img[i, j, 0] = (
                        bayer_img[i - 1, j, 2] / 2.0 + bayer_img[i + 1, j, 2] / 2.0)
                    rgb_img[i, j, 2] = (
                        bayer_img[i, j - 1, 0] / 2.0 + bayer_img[i, j + 1, 0] / 2.0)
                else:
                    rgb_img[i, j, 0] = (bayer_img[i - 1, j - 1, 0] / 4.0 + bayer_img[i - 1, j + 1, 0] /
                                        4.0 + bayer_img[i + 1, j - 1, 0] / 4.0 + bayer_img[i + 1, j + 1, 2] / 4.0)
                    rgb_img[i, j, 1] = (bayer_img[i - 1, j, 1] / 4.0 + bayer_img[i, j + 1, 1] /
                                        4.0 + bayer_img[i, j - 1, 1] / 4.0 + bayer_img[i + 1, j, 1] / 4.0)

    rgb_img[:, [0, w - 1], :] = 0
    rgb_img[[0, h - 1], :, :] = 0

    return rgb_img


bayer_img = cv2.imread('Imagens/Lighthouse_bayerBG8.png')

rgb_img = bayer_para_rgb(bayer_img)

cv2.imshow('Imagem de Entrada', bayer_img)
cv2.imshow('Imagem de Saida (com bayer)', rgb_img)
cv2.imwrite('LighthouseQ2.png', rgb_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

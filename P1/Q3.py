import cv2
import numpy as np


def rgb_to_gray(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)


def isodata_threshold(gray_img):
    t = np.mean(gray_img)
    while True:
        g1 = gray_img[gray_img > t]
        g2 = gray_img[gray_img <= t]

        if g1.size == 0 or g2.size == 0:
            break

        new_t = 0.5 * (np.mean(g1) + np.mean(g2))

        if abs(new_t - t) < 0.5:
            return int(new_t)

        t = new_t


img = cv2.imread('LighthouseQ2.png')
gray_img = rgb_to_gray(img)
threshold = isodata_threshold(gray_img)
_, binary_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

cv2.imshow('Imagem de Entrada', img)
cv2.imshow('Imagem de Saida', binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

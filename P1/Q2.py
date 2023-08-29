import cv2
import numpy as np

def demosaic_bayer(image_path):
    # Ler a imagem em escala de cinza
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Adicionar uma moldura de zeros ao redor da imagem para tratar os pixels nas bordas
    img_padded = cv2.copyMakeBorder(img_gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # Inicializar uma matriz vazia para armazenar a imagem RGB
    img_rgb = np.zeros((img_padded.shape[0], img_padded.shape[1], 3), dtype=np.uint8)

    # Interpolação
    for i in range(1, img_padded.shape[0] - 1):
        for j in range(1, img_padded.shape[1] - 1):
            if (i % 2 == 0 and j % 2 == 0):  # Posição azul
                img_rgb[i, j, 0] = img_padded[i, j]
                img_rgb[i, j, 1] = (img_padded[i-1, j] + img_padded[i+1, j] + img_padded[i, j-1] + img_padded[i, j+1]) // 4
                img_rgb[i, j, 2] = (img_padded[i-1, j-1] + img_padded[i-1, j+1] + img_padded[i+1, j-1] + img_padded[i+1, j+1]) // 4
            elif (i % 2 != 0 and j % 2 != 0):  # Posição vermelha
                img_rgb[i, j, 0] = (img_padded[i-1, j-1] + img_padded[i-1, j+1] + img_padded[i+1, j-1] + img_padded[i+1, j+1]) // 4
                img_rgb[i, j, 1] = (img_padded[i-1, j] + img_padded[i+1, j] + img_padded[i, j-1] + img_padded[i, j+1]) // 4
                img_rgb[i, j, 2] = img_padded[i, j]
            else:  # Posição verde
                img_rgb[i, j, 0] = (img_padded[i-1, j] + img_padded[i+1, j]) // 2 if i % 2 == 0 else (img_padded[i, j-1] + img_padded[i, j+1]) // 2
                img_rgb[i, j, 1] = img_padded[i, j]
                img_rgb[i, j, 2] = (img_padded[i, j-1] + img_padded[i, j+1]) // 2 if i % 2 == 0 else (img_padded[i-1, j] + img_padded[i+1, j]) // 2

    # Remover a moldura de zeros
    img_rgb = img_rgb[1:-1, 1:-1]

    return img_rgb

# Teste do script
if __name__ == "__main__":
    image_path = "Imagens/Lighthouse_bayerBG8.png"  # Substitua pelo caminho da sua imagem
    img_rgb_result = demosaic_bayer(image_path)
    
    # Exibir o resultado
    cv2.imshow('Resultado da Interpolacao Bayer', img_rgb_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

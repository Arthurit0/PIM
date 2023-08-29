import cv2
import numpy as np


def identificar_cor(imagem_path):
    # Ler a imagem
    imagem = cv2.imread(imagem_path)

    # Calcular o histograma de cores
    histograma_r = cv2.calcHist([imagem], [0], None, [256], [0, 256])
    histograma_g = cv2.calcHist([imagem], [1], None, [256], [0, 256])
    histograma_b = cv2.calcHist([imagem], [2], None, [256], [0, 256])

    # Obter a cor predominante em cada canal
    r_predominante = np.argmax(histograma_r)
    g_predominante = np.argmax(histograma_g)
    b_predominante = np.argmax(histograma_b)

    # Identificar a cor do carro
    if r_predominante > 200 and g_predominante > 200 and b_predominante > 200:
        return "branco"
    elif r_predominante < 50 and g_predominante < 50 and b_predominante < 50:
        return "preto"
    elif r_predominante > g_predominante and r_predominante > b_predominante:
        return "vermelho"
    elif g_predominante > r_predominante and g_predominante > b_predominante:
        return "verde"
    elif b_predominante > r_predominante and b_predominante > g_predominante:
        return "azul"
    else:
        return "Indeterminada"


# Teste do script
if __name__ == "__main__":
    for cor in ["white", "black", "green", "red", "blue"]:
        imagem_path = f"Imagens/{cor}.png"  # Substitua pelo caminho da sua imagem
        identificaCor = identificar_cor(imagem_path)
        print(f"A cor do carro é {identificaCor}.")

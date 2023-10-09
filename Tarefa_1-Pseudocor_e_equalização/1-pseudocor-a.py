import numpy as np
import cv2
import random

def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]

def main():
    mapa = cv2.imread('Imgs_Originais/mapa.png', cv2.IMREAD_GRAYSCALE)
    taxa_roubo = cv2.imread('Imgs_Originais/taxaPerCapitaRouboCarros.png', cv2.IMREAD_GRAYSCALE)

    if mapa.shape != taxa_roubo.shape:
        print("As dimensões das imagens devem ser iguais.")
        return

    result_image = np.zeros((mapa.shape[0], mapa.shape[1], 3), dtype=np.uint8)  # Fundo preto

    unique_values = np.unique(taxa_roubo)
    color_mapping = {}

    for value in unique_values:
        if value == 0:  # Ignorar o preto (fronteiras) e o fundo branco
            continue
        color_mapping[value] = generate_random_color()

    for value, color in color_mapping.items():
        mask = (taxa_roubo == value)
        for i in range(3):
            result_image[:,:,i][mask] = color[i]

    cv2.imwrite('mapa_pseudocolor_regioes.png', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    resized_image = cv2.resize(result_image, (600, 600))

    # Exibir a imagem no formato RGB
    cv2.imshow('Mapa Pseudocor Regioes (qualquer tecla para fechar)', cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

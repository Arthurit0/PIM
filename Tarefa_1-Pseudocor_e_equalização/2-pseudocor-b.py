import numpy as np
import cv2

img_values = set()

def map_gray_to_color(value, limiar):
    mapped_value = 300 - (value / 255.0) * 300
    unique_values = np.unique(np.round(mapped_value, 2))
    img_values.update(unique_values.tolist())

    color_map = np.zeros((value.shape[0], value.shape[1], 3), dtype=np.uint8)

    mask = mapped_value > limiar
    # Aplica vermelho onde a condição da máscara é verdadeira
    color_map[mask] = [0, 0, 255]
    # Aplica cinza onde a condição da máscara é falsa
    color_map[~mask] = np.stack([value[~mask], value[~mask], value[~mask]], axis=-1)

    return color_map

def main():
    limiar = -1

    while limiar < 1 or limiar > 300:
        limiar = int(input('Valor do limiar? (Entre 1 e 300) '))
        if limiar < 1 or limiar > 300:
            print("Valor inválido! Tente novamente...")

    mapa = cv2.imread('Imgs_Originais/mapa.png', cv2.IMREAD_GRAYSCALE)
    taxa_roubo = cv2.imread('Imgs_Originais/taxaPerCapitaRouboCarros.png', cv2.IMREAD_GRAYSCALE)

    if mapa.shape != taxa_roubo.shape:
        print("As dimensões das imagens devem ser iguais.")
        return
    
    # Cria uma máscara para pixels com valores 0 ou 255
    mask = np.logical_or(taxa_roubo == 0, taxa_roubo == 255)
    result_image = map_gray_to_color(taxa_roubo, limiar)
    result_image[mask] = np.stack([taxa_roubo[mask], taxa_roubo[mask], taxa_roubo[mask]], axis=-1)

    # Salvar a imagem no formato BGR
    cv2.imwrite('mapa_pseudocolor.png', result_image)
    print(f'Valores de cinza: {sorted(list(img_values))}')

    # Exibir a imagem usando OpenCV
    cv2.imshow('Resultado (Qualquer tecla para fechar)', cv2.resize(result_image, (600, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

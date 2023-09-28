import numpy as np
import cv2

img_values = []

def map_gray_to_color(value, limiar):
    mapped_value = 300 - (value / 255.0) * 300
    
    if mapped_value not in img_values:
        img_values.append(round(mapped_value, 2)) 

    if mapped_value > limiar:
        # BGR
        return [0, 0, 255]
    else:
        return [value, value, value]

def main():
    limiar = -1

    while limiar < 1 or limiar > 300:
        limiar = int(input('Valor do limiar? (Entre 1 e 300) '))
        if limiar < 1 or limiar > 300:
            print("Valor inválido! Tente novamente...")

    mapa = cv2.imread('./Imgs_Originais/mapa.png', cv2.IMREAD_GRAYSCALE)
    taxa_roubo = cv2.imread('./Imgs_Originais/taxaPerCapitaRouboCarros.png', cv2.IMREAD_GRAYSCALE)

    if mapa.shape != taxa_roubo.shape:
        print("As dimensões das imagens devem ser iguais.")
        return

    result_image = np.zeros((mapa.shape[0], mapa.shape[1], 3), dtype=np.uint8)

    for y in range(mapa.shape[0]):
        for x in range(mapa.shape[1]):
            value = taxa_roubo[y, x]
            if value == 0 or value == 255:
                map_val = [value, value, value]
            else:
                map_val = map_gray_to_color(value, limiar)
            result_image[y, x] = map_val

    # Salvar a imagem no formato BGR
    cv2.imwrite('mapa_pseudocolor.png', result_image)
    print(f'Valores de cinza: {img_values}')

    # Exibir a imagem usando OpenCV
    cv2.imshow('Resultado', cv2.resize(result_image, (600, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np

imagens = ['Lua1_gray.jpg', 'chessboard_inv.png', 'img02.jpg']

operadores = {
    'prewitt': {
        'x': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        'y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    },
    'sobel': {
        'x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    },
    'scharr': {
        'x': np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]),
        'y': np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
    }
}

def gera_histograma(direction, magnitude):
    bins = np.arange(0, 181, 20)
    histogram, _ = np.histogram(direction, bins=bins, weights=magnitude, density=True)
    return histogram

def salvar_histogramas(histogram, image_name, op_name):
    with open(f'./histogramas/{image_name[:-4]}_{op_name}.txt', 'w') as f:
        f.write(f'Histograma para {image_name} usando {op_name}:\n')
        for i in range(9):
            f.write(f'{i*20}-{(i+1)*20}: {histogram[i]:.4f}\n')

def operador_gradiente(img, operator_x, operator_y):
    g_x = cv2.filter2D(img, -1, operator_x)
    g_y = cv2.filter2D(img, -1, operator_y)
    
    magnitude = np.sqrt(g_x**2 + g_y**2)

    if np.any(np.isnan(magnitude)) or np.any(np.isinf(magnitude)):
        print("NaN ou Infinito!")
        magnitude[np.isnan(magnitude)] = 0
        magnitude[np.isinf(magnitude)] = 0

    magnitude = magnitude.astype(np.float32)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    direcao = np.arctan2(g_y, g_x) * (180/np.pi)
    direcao = (direcao + 180) % 180
    
    return magnitude, direcao

if __name__ == '__main__':
    for imagem in imagens:
        img_path = f'./orig_imgs/{imagem}'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        for nome_op, mascaras_op in operadores.items():
            magnitude, direcao = operador_gradiente(img, mascaras_op['x'], mascaras_op['y'])

            histograma = gera_histograma(direcao, magnitude)
            salvar_histogramas(histograma, imagem, nome_op)

            cv2.imwrite(f'./result_imgs/magnitude/{imagem[:-4]}_{nome_op}.png', magnitude)
            cv2.imwrite(f'./result_imgs/direction/{imagem[:-4]}_{nome_op}.png', direcao)

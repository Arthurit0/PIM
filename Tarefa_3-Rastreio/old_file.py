import cv2
import numpy as np
import csv
import os

frames_dir = './frames'  # Substitua pelo seu diretório de quadros
template_filename = 'im1.jpg'  # Nome do arquivo do template

# Carregar o template e determinar seu tamanho
template = cv2.imread(os.path.join(frames_dir, template_filename), 0)
h, w = template.shape[:2]

# Métodos de Template Matching
methods = [
    'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
    'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
    'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
]

# Processar cada método
for method in methods:
    results = []
    marked_frames = []

    # Processar cada quadro
    for frame_filename in sorted(os.listdir(frames_dir)):
        if frame_filename.endswith(('.jpg', '.jpeg', '.png')):  # Verifica o formato do arquivo
            frame_path = os.path.join(frames_dir, frame_filename)
            frame = cv2.imread(frame_path, 0)
            method_eval = eval(method)
            res = cv2.matchTemplate(frame, template, method_eval)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if method in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # Marcar o objeto em cada quadro
            marked_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Converter para BGR para marcar com cor
            cv2.rectangle(marked_frame, top_left, bottom_right, (0, 0, 255), 2)  # Marcar com retângulo vermelho
            marked_frames.append(marked_frame)

            # Adicionar resultados na lista
            results.append([frame_filename, min_val, max_val])

    # Salvar resultados em CSV
    with open(f'{method}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Quadro', 'Min_Val', 'Max_Val'])
        writer.writerows(results)

    # Criar um vídeo de saída para o método atual
    out = cv2.VideoWriter(f'rastreado_{method}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (w, h))
    for marked_frame in marked_frames:
        out.write(marked_frame)
    out.release()

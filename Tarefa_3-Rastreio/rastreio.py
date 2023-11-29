import cv2
import csv
import os
import matplotlib.pyplot as plt

frames_dir = './frames'
template_filename = 'template.png'

template = cv2.imread(template_filename, 0)
template_height, template_width = template.shape[:2]

first_frame_path = './frames/frame0001.png'
first_frame = cv2.imread(first_frame_path, 0)
video_height, video_width = first_frame.shape[:2]

methods = [
    'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
    'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
    'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
]

for method in methods:
    results = []
    marked_frames = []

    print(f"Processando com o método: {method}")
    
    for frame_filename in sorted(os.listdir(frames_dir)):
        frame = cv2.imread(f'{frames_dir}/{frame_filename}', 0)
        res = cv2.matchTemplate(frame, template, eval(method))
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
            top_left = min_loc
        else:
            top_left = max_loc
            
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

        marked_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  
        cv2.rectangle(marked_frame, top_left, bottom_right, (0, 0, 255), 2)  
        marked_frames.append(marked_frame)

        results.append([frame_filename, min_val, max_val])

    with open(f'res_rastreio/tabelas/{method[4:]}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Quadro', 'Min_Val', 'Max_Val'])
        writer.writerows(results)

    out = cv2.VideoWriter(f'res_rastreio/videos/{method[4:]}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (video_width, video_height))
    for marked_frame in marked_frames:
        out.write(marked_frame)
    out.release()

for method in methods:
    quadros = []
    min_vals = []
    max_vals = []

    with open(os.path.join('res_rastreio/tabelas', f'{method[4:]}.csv'), newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            quadro, min_val, max_val = row
            quadros.append(quadro[6:9])
            min_vals.append(float(min_val))
            max_vals.append(float(max_val))

    plt.figure(figsize=(10, 6))
    plt.plot(quadros, min_vals, label='Min_Val')
    plt.plot(quadros, max_vals, label='Max_Val')
    plt.title(f'Resultados do Método {method[4:]}')
    plt.xlabel('Quadros')
    plt.ylabel('Valor')
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.grid(False)
    plt.legend()
    plt.savefig(f'res_rastreio/graficos/{method[4:]}.png')

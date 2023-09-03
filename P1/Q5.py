import numpy as np


def matriz_calibracao(pontos):
    A = []

    for X, Y, Z, x, y in pontos:
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])

    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]

    H = h.reshape(3, 4)

    return H


def pontos_projetados(H, pontos):
    pontos_proj = []

    for X, Y, Z, _, _ in pontos:
        p = np.dot(H, np.array([X, Y, Z, 1]))
        x, y, w = p[0], p[1], p[2]
        pontos_proj.append((x/w, y/w))

    return pontos_proj


def calcula_acuracia(pontos_orig, pontos_proj):
    erro = 0

    for (x1, y1), (x2, y2) in zip(pontos_orig, pontos_proj):
        erro += np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return erro / len(pontos_orig)


pontos = [
    (0.0, 1000.0, 200.0, 5, 3),
    (600.0, 1000.0, 400.0, 3, 4),
    (1000.0, 800.0, 400.0, 2, 4),
    (1000.0, 400.0, 600.0, 1, 2),
    (1000.0, 1000.0, 600.0, 2, 3),
    (1000.0, 800.0, 0.0, 2, 5)
]

H = matriz_calibracao(pontos)

pontos_proj = pontos_projetados(H, pontos)

pontos_orig_2d = [(x, y) for _, _, _, x, y in pontos]
acuracia = calcula_acuracia(pontos_orig_2d, pontos_proj)

print(f"Matriz de calibração: \n{H}")
print(f"Acurácia: {acuracia}")

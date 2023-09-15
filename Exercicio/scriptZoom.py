import cv2
import numpy as np

def exibir_regiao(imagem, i, j, dimensao_janela):
    borda = dimensao_janela // 2
    imagem_auxiliar = cv2.copyMakeBorder(imagem, borda, borda, borda, borda, cv2.BORDER_REFLECT)
    
    regiao = imagem_auxiliar[i:i+dimensao_janela, j:j+dimensao_janela]

    media = np.mean(regiao)
    desvio_padrao = np.std(regiao)

    vizinhos = ""
    for x in range(i, i+dimensao_janela):
        for y in range(j, j+dimensao_janela):
            if (x, y) != (i+borda, j+borda):
                vizinhos += f'({x}, {y})\n'

    # Exibindo a região
    cv2.imshow('Regiao', regiao)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    with open('saida_exercicio.txt', 'w') as saida:
        saida.write(f"Média: {media}\nDesvio padrão: {desvio_padrao}\nVizinhos: {vizinhos}")

if __name__ == '__main__':
    # Carregando uma imagem em tons de cinza
    imagem = cv2.imread('./marilyn.jpg', cv2.IMREAD_GRAYSCALE)

    # Exibindo a altura e a largura da imagem
    altura, largura = imagem.shape
    print("Dados da Imagem:\n")
    print(f"- Altura: {altura}")
    print(f"- Largura: {largura}\n")

    while True:
        # Solicitando as coordenadas e a dimensão da janela do usuário
        i = int(input("Digite o valor de i: "))
        j = int(input("Digite o valor de j: "))
        dimensao_janela = int(input("Digite a dimensão da janela (deve ser ímpar): "))

        # Verificando se os valores inseridos são válidos
        if dimensao_janela % 2 == 0:
            print("A dimensão da janela deve ser ímpar. Tente novamente.")
            continue

        if i < 0 or i >= altura or j < 0 or j >= largura or dimensao_janela <= 0 or i + dimensao_janela > altura or j + dimensao_janela > largura:
            print("\nValores inválidos inseridos (janela fora das dimensões da imagem). Tente novamente.\n")
            continue
        
        # Chamando a função com as coordenadas (i, j) e a dimensão da janela
        exibir_regiao(imagem, i, j, dimensao_janela)
        break

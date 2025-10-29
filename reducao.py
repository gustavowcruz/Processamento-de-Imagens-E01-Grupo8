import cv2
import numpy as np
import os

NOME_ARQUIVO_ENTRADA = 'content/images.jpg'

NOME_ARQUIVO_SAIDA_GAUSSIAN = 'resultado_gaussian_blur.png'
NOME_ARQUIVO_SAIDA_MEDIAN = 'resultado_median_blur.png'
NOME_ARQUIVO_SAIDA_NL_MEANS = 'resultado_nl_means.png'


def reduzirRuido():
    print(f"Processando a imagem: {NOME_ARQUIVO_ENTRADA}")

    if not os.path.exists(NOME_ARQUIVO_ENTRADA):
        print(f"ERRO: Arquivo não encontrado em '{NOME_ARQUIVO_ENTRADA}'")
        print("Por favor, coloque a imagem no mesmo diretório do script ou atualize o caminho.")
    else:
        img = cv2.imread(NOME_ARQUIVO_ENTRADA)

        if img is None:
            print("ERRO: Não foi possível ler a imagem. Verifique se o arquivo está corrompido.")
        else:
            print("Imagem carregada com sucesso. Aplicando filtros...")

            img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

            img_median = cv2.medianBlur(img, 5)

            img_nl_means = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


            try:
                cv2.imwrite(NOME_ARQUIVO_SAIDA_GAUSSIAN, img_gaussian)
                cv2.imwrite(NOME_ARQUIVO_SAIDA_MEDIAN, img_median)
                cv2.imwrite(NOME_ARQUIVO_SAIDA_NL_MEANS, img_nl_means)

                print("\n--- Processamento Concluído ---")
                print(f"Output 1 (Gaussiano): {NOME_ARQUIVO_SAIDA_GAUSSIAN}")
                print(f"Output 2 (Mediana):    {NOME_ARQUIVO_SAIDA_MEDIAN}")
                print(f"Output 3 (NL-Means):   {NOME_ARQUIVO_SAIDA_NL_MEANS}")

            except Exception as e:
                print(f"ERRO ao salvar as imagens: {e}")

            print("\nExibindo resultados... Pressione 'q' em qualquer janela para sair.")
            cv2.imshow('Original', img)
            cv2.imshow('NL Means (Recomendado)', img_nl_means)
            cv2.imshow('Median', img_median)

            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()

reduzirRuido()
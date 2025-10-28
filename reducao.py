import cv2
import numpy as np
import os
# import matplotlib.pyplot as plt

# --- Configuração ---
# Coloque o nome da sua imagem de entrada aqui
NOME_ARQUIVO_ENTRADA = 'content/images.jpg'

# Nomes dos arquivos de saída
NOME_ARQUIVO_SAIDA_GAUSSIAN = 'resultado_gaussian_blur.png'
NOME_ARQUIVO_SAIDA_MEDIAN = 'resultado_median_blur.png'
NOME_ARQUIVO_SAIDA_NL_MEANS = 'resultado_nl_means.png'

# --- 1. Ler a Imagem ---

print(f"Processando a imagem: {NOME_ARQUIVO_ENTRADA}")

# Verifica se o arquivo existe antes de tentar carregar
if not os.path.exists(NOME_ARQUIVO_ENTRADA):
    print(f"ERRO: Arquivo não encontrado em '{NOME_ARQUIVO_ENTRADA}'")
    print("Por favor, coloque a imagem no mesmo diretório do script ou atualize o caminho.")
else:
    # Carrega a imagem colorida
    img = cv2.imread(NOME_ARQUIVO_ENTRADA)

    if img is None:
        print("ERRO: Não foi possível ler a imagem. Verifique se o arquivo está corrompido.")
    else:
        print("Imagem carregada com sucesso. Aplicando filtros...")

        # --- 2. Aplicar Redução de Ruído ---

        # Método 1: Gaussian Blur (Desfoque Gaussiano)
        # Bom para ruído Gaussiano leve.
        # (img, kernel_size, sigmaX) - O kernel 5x5 é um bom ponto de partida.
        img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

        # Método 2: Median Blur (Filtro de Mediana)
        # Excelente para ruído "Sal e Pimenta" (pontos brancos/pretos aleatórios).
        # (img, kernel_size) - O kernel deve ser um ímpar.
        img_median = cv2.medianBlur(img, 5)

        # Método 3: Non-Local Means (NL-Means)
        # O método mais avançado e eficaz para ruído geral, preservando melhor as bordas.
        # É mais lento, mas oferece resultados superiores.
        # (img, dest, h, hColor, templateWindowSize, searchWindowSize)
        # 'h' (e 'hColor') é o parâmetro de força do filtro. 10 é um bom valor inicial.
        img_nl_means = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # --- 3. Salvar as Imagens (Output) ---

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

        # Opcional: Mostrar as imagens na tela (pressione 'q' para fechar)
        print("\nExibindo resultados... Pressione 'q' em qualquer janela para sair.")
        cv2.imshow('Original', img)
        cv2.imshow('NL Means (Recomendado)', img_nl_means)
        cv2.imshow('Median', img_median)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
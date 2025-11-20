● Integrante: Alan Reis Anjos
● Módulo/Técnica Principal: Análise de Vizinhança e Segmentação
● Funções Específicas a Desenvolver: segment_text_lines(binary_image): Função que identifica e separa linhas de texto em uma imagem. get_neighbors(pixel, connectivity_type): Retorna os vizinhos de um pixel com base na conectividade 4 ou 8.
● Responsabilidade no Relatório: Documentar a metodologia de segmentação de linhas e a análise de vizinhança.

● Integrante: Gabriel Lopes dos Santos
● Módulo/Técnica Principal: Segmentação por Watershed
● Funções Específicas a Desenvolver: apply_watershed(image): Implementa o algoritmo de watershed para separar objetos conectados. generate_markers(image): Cria os marcadores iniciais para o watershed.
● Responsabilidade no Relatório: Explicar o funcionamento e a aplicação do algoritmo de Watershed no contexto do projeto.

● Integrante: Gustavo Wagner Cruz Cunha
● Módulo/Técnica Principal: Pré-processamento e Filtragem
● Funções Específicas a Desenvolver: apply_morphological_filter(image, filter_type): Aplica filtros de abertura e fechamento para remoção de ruído. binarize_image(image, threshold): Converte a imagem para preto e branco.
● Responsabilidade no Relatório: Descrever as técnicas de pré-processamento e justificar a escolha dos filtros morfológicos.

● Integrante: João Damasceno Ferraz Neto
● Módulo/Técnica Principal: Extração de Texto (OCR)
● Funções Específicas a Desenvolver: preprocess_for_ocr(image): Aplica uma sequência de filtros para otimizar a imagem para o reconhecimento de caracteres. extract_text(image_component): Utiliza uma biblioteca de OCR para extrair texto de um componente isolado.
● Responsabilidade no Relatório: Escrever os casos de uso UC02 e UC04, detalhando o fluxo de OCR.

● Integrante: Luís Felipe Castelo Branco do Vale
● Módulo/Técnica Principal: Transformada de Distância e Métricas
● Funções Específicas a Desenvolver: calculate_distance_transform(binary_image): Gera o mapa de distância da imagem. extract_metric (labeled_image): Calcula área, perímetro e centróide para cada componente rotulado.
● Responsabilidade no Relatório: Elaborar a seção sobre métricas de componentes e a aplicação da transformada de distância.

REQUISITOS

----------------
ADCIONAR DATASET
90kDICT32px NA PASTA DATASETS 


Projeto completo com o dataset:
https://drive.google.com/file/d/1cT2yY2tiNi_V39r3HPTC5DTY2GlotYO1/view?usp=sharing

Rode pip install -r requiriments.txt no terminal para instalar as bibliotecas necessárias.

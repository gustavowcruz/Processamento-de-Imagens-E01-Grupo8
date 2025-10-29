import cv2
import numpy as np
from matplotlib import pyplot as plt

# --- Simulação ---

# Módulo de Pré-processamento
# Criar uma imagem binária com objetos sobrepostos
def criar_imagem_binaria_exemplo():
    # Cria uma imagem preta
    img = np.zeros((300, 300), dtype=np.uint8)
    # Desenha dois círculos brancos que se sobrepõem
    cv2.circle(img, (120, 150), 70, 255, -1)
    cv2.circle(img, (180, 150), 70, 255, -1)
    return img

# Módulo de Transformada de Distância
# Dependência direta para a geração de marcadores
def calculate_distance_transform(binary_image):
    #Calcula a Transformada de Distância Euclidiana (DIST_L2) de uma imagem binária
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
    return dist_transform

# Módulo de Segmentação por Watershed
def generate_markers(binary_image):
    #Cria os marcadores (sementes) para o algoritmo Watershed

    
    # 1. Calcular a Transformada de Distância
    # Isso nos dá os "centros" dos objetos 
    dist_transform = calculate_distance_transform(binary_image)
    
    # 2. Identificar "Primeiro Plano Certo" (Sure Foreground)
    # Aplica um limiar na transformada de distância
    # Pixels com alta distância (claros) são objetos
    # O valor 0.7*max() é um ponto de partida
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 3. Identificar "Fundo Certo" (Sure Background)
    # Dilata a imagem original. A área que cresce é o fundo
    sure_bg = cv2.dilate(binary_image, None, iterations=3)
    
    # 4. Identificar Região Desconhecida
    # A região que não é nem fundo certo nem primeiro plano certo
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 5. Criar Mapa de Marcadores
    # Rotula os componentes do "primeiro plano certo" 
    # O OpenCV connectedComponents rotula o fundo como 0
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Adiciona 1 a todos os rótulos para que 0 (fundo) não seja um rótulo
    # Agora, 0 é "desconhecido", 1 é "fundo", 2+ são objetos
    markers = markers + 1
    
    # Marca a região desconhecida (calculada na Etapa 4) com 0
    markers[unknown == 255] = 0
    
    # Retorna os marcadores e imagens intermediárias para visualização
    return markers, dist_transform, sure_fg, sure_bg

# Aplica o algoritmo Watershed para segmentar a imagem.
def apply_watershed(original_image, markers):
    
    # O algoritmo Watershed do OpenCV espera uma imagem colorida (3 canais)
    # Se a imagem for binária, converta-a para BGR
    if len(original_image.shape) == 2:
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        original_image_color = original_image.copy()

    # Aplica o Watershed
    # O algoritmo modifica o array 'markers'
    cv2.watershed(original_image_color, markers)
    
    # As bordas (cumes) que separam os objetos são marcadas com -1
    # Pinta as bordas de vermelho na imagem colorida
    original_image_color[markers == -1] = [0, 0, 255] # Formato BGR
    
    # 'markers' agora contém os objetos segmentados (rótulos > 1) e as bordas (-1)
    return original_image_color, markers


# Execução do Fluxo Principal

# 1. Obter a imagem pré-processada (binarizada)
binary_image = criar_imagem_binaria_exemplo()

# 2. Gerar os marcadores
markers, dist_transform, sure_fg, sure_bg = generate_markers(binary_image)

# 3. Aplicar o Watershed
# Utiliza a imagem binária 'original_image' para este exemplo
segmented_image_visual, segmented_markers = apply_watershed(binary_image, markers)

# 4. Integração com 'segmented_markers' é enviado para a função 'extract_metrics'


# Visualização completa do Módulo Watershed
plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(binary_image, cmap='gray')
plt.title('1. Imagem Binária')

plt.subplot(232)
plt.imshow(dist_transform, cmap='gray')
plt.title('2. Transformada de Distância')

plt.subplot(233)
plt.imshow(sure_fg, cmap='gray')
plt.title('3. Marcadores: "Primeiro Plano"')

plt.subplot(234)
plt.imshow(sure_bg, cmap='gray')
plt.title('4. Marcadores: "Fundo"')

plt.subplot(235)
plt.imshow(markers, cmap='jet') # 'jet' melhora a visualização de rótulos
plt.title('5. Marcadores Finais (0 = Desconhecido)')

plt.subplot(236)
plt.imshow(cv2.cvtColor(segmented_image_visual, cv2.COLOR_BGR2RGB))
plt.title('6. Resultado do Watershed')

plt.tight_layout()
plt.show()

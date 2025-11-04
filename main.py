<<<<<<< HEAD
import numpy as np

def get_neighbors(pixel, image_shape, connectivity_type=8):
    """
    Retorna os vizinhos de um pixel com base na conectividade 4 ou 8.

    Args:
        pixel (tuple): As coordenadas (linha, coluna) do pixel.
        image_shape (tuple): As dimensões (altura, largura) da imagem.
        connectivity_type (int): O tipo de conectividade (4 ou 8).

    Returns:
        list: Uma lista de tuplas com as coordenadas dos vizinhos.
    """
    (row, col) = pixel
    (height, width) = image_shape
    neighbors = []

    if connectivity_type == 4:
        # Vizinhança 4: cima, baixo, esquerda, direita
        if row > 0: neighbors.append((row - 1, col))
        if row < height - 1: neighbors.append((row + 1, col))
        if col > 0: neighbors.append((row, col - 1))
        if col < width - 1: neighbors.append((row, col + 1))
    elif connectivity_type == 8:
        # Vizinhança 8: inclui diagonais
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                n_row, n_col = row + i, col + j
                if 0 <= n_row < height and 0 <= n_col < width:
                    neighbors.append((n_row, n_col))
    else:
        raise ValueError("O tipo de conectividade deve ser 4 ou 8.")

    return neighbors

def segment_text_lines(binary_image):
    """
    Identifica e separa linhas de texto em uma imagem binária.
    Esta é uma implementação básica que usa projeção horizontal.

    Args:
        binary_image (np.array): Uma imagem binária (0 para fundo, 1 para texto).

    Returns:
        list: Uma lista de imagens, onde cada imagem é uma linha de texto segmentada.
    """
    if not isinstance(binary_image, np.ndarray):
        raise TypeError("A imagem de entrada deve ser um array numpy.")

    # Projeção horizontal: soma dos valores dos pixels em cada linha
    horizontal_projection = np.sum(binary_image, axis=1)

    # Encontra as linhas onde a projeção é não-nula (contêm texto)
    non_zero_lines = np.where(horizontal_projection > 0)[0]

    if len(non_zero_lines) == 0:
        return []

    # Encontra os inícios e fins das linhas de texto
    line_breaks = np.where(np.diff(non_zero_lines) > 1)[0]
    line_starts = [non_zero_lines[0]]
    if len(line_breaks) > 0:
        line_starts.extend(non_zero_lines[line_breaks + 1])

    line_ends = []
    if len(line_breaks) > 0:
        line_ends.extend(non_zero_lines[line_breaks])
    line_ends.append(non_zero_lines[-1])


    # Recorta as linhas da imagem original
    segmented_lines = []
    for start, end in zip(line_starts, line_ends):
        # Adiciona um pequeno preenchimento para garantir que a linha inteira seja capturada
        start_row = max(0, start - 1)
        end_row = min(binary_image.shape[0], end + 2)
        line_image = binary_image[start_row:end_row, :]
        segmented_lines.append(line_image)

    return segmented_lines

# Funções existentes
def ler(imagem):
    return None

def exibir(imagem):
    return None

# Exemplo de uso (pode ser removido ou adaptado)
if __name__ == '__main__':
    # Criar uma imagem de exemplo
    example_image = np.zeros((50, 200), dtype=int)
    example_image[10:15, 10:190] = 1 # Linha 1
    example_image[25:30, 10:190] = 1 # Linha 2
    example_image[40:45, 10:190] = 1 # Linha 3

    print("Imagem de exemplo criada com forma:", example_image.shape)

    # Testar segmentação de linhas
    lines = segment_text_lines(example_image)
    print(f"Número de linhas de texto segmentadas: {len(lines)}")
    for i, line in enumerate(lines):
        print(f"  - Linha {i+1} com forma: {line.shape}")


    # Testar vizinhança
    pixel_to_test = (10, 10)
    image_shape_for_test = example_image.shape
    print(f"\nTestando vizinhos para o pixel {pixel_to_test}:")
    neighbors_4 = get_neighbors(pixel_to_test, image_shape_for_test, connectivity_type=4)
    print(f"  - Vizinhança 4: {neighbors_4}")
    neighbors_8 = get_neighbors(pixel_to_test, image_shape_for_test, connectivity_type=8)
    print(f"  - Vizinhança 8: {neighbors_8}")
=======
print('hello world')
>>>>>>> main

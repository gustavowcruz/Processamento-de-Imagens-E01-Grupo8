import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
from collections import defaultdict, deque
import pandas as pd
from skimage import measure, morphology
from skimage.util import random_noise
import seaborn as sns
from typing import Tuple, List, Dict

class AnalisadorConectividade:
    def __init__(self):
        self.resultados = {}
    
    def binarizar_imagem(self, imagem: np.ndarray, limiar: int = 128) -> np.ndarray:
        """Converte imagem para binária usando limiarização"""
        if len(imagem.shape) == 3:
            imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        _, binaria = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY)
        return binaria
    
    def adicionar_ruido(self, imagem: np.ndarray, tipo_ruido: str = 'salt', quantidade: float = 0.05) -> np.ndarray:
        """Adiciona ruído à imagem binária"""
        ruidosa = random_noise(imagem/255.0, mode=tipo_ruido, amount=quantidade)
        return (ruidosa * 255).astype(np.uint8)
    
    def rotulagem_vizinhanca_4(self, imagem: np.ndarray) -> Tuple[np.ndarray, int]:
        """Implementa rotulagem de componentes usando vizinhança-4"""
        altura, largura = imagem.shape
        rotulada = np.zeros((altura, largura), dtype=np.int32)
        rotulo_atual = 1
        equivalencias = {}
        
        # Primeira passada
        for i in range(altura):
            for j in range(largura):
                if imagem[i, j] == 255:  # Pixel foreground
                    vizinhos = []
                    if i > 0 and rotulada[i-1, j] > 0:
                        vizinhos.append(rotulada[i-1, j])
                    if j > 0 and rotulada[i, j-1] > 0:
                        vizinhos.append(rotulada[i, j-1])
                    
                    if not vizinhos:
                        rotulada[i, j] = rotulo_atual
                        rotulo_atual += 1
                    else:
                        rotulo_min = min(vizinhos)
                        rotulada[i, j] = rotulo_min
                        # Registrar equivalências
                        for vizinho in vizinhos:
                            if vizinho != rotulo_min:
                                equivalencias[vizinho] = rotulo_min
        
        # Resolver equivalências
        return self._resolver_equivalencias(rotulada, equivalencias, rotulo_atual)
    
    def rotulagem_vizinhanca_8(self, imagem: np.ndarray) -> Tuple[np.ndarray, int]:
        """Implementa rotulagem de componentes usando vizinhança-8"""
        altura, largura = imagem.shape
        rotulada = np.zeros((altura, largura), dtype=np.int32)
        rotulo_atual = 1
        equivalencias = {}
        
        # Primeira passada
        for i in range(altura):
            for j in range(largura):
                if imagem[i, j] == 255:  # Pixel foreground
                    vizinhos = []
                    # Vizinhos na vizinhança-8
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < altura and 0 <= nj < largura and rotulada[ni, nj] > 0:
                                vizinhos.append(rotulada[ni, nj])
                    
                    if not vizinhos:
                        rotulada[i, j] = rotulo_atual
                        rotulo_atual += 1
                    else:
                        rotulo_min = min(vizinhos)
                        rotulada[i, j] = rotulo_min
                        # Registrar equivalências
                        for vizinho in vizinhos:
                            if vizinho != rotulo_min:
                                equivalencias[vizinho] = rotulo_min
        
        # Resolver equivalências
        return self._resolver_equivalencias(rotulada, equivalencias, rotulo_atual)
    
    def _resolver_equivalencias(self, rotulada: np.ndarray, equivalencias: Dict, max_rotulo: int) -> Tuple[np.ndarray, int]:
        """Resolve equivalências entre rótulos"""
        # Criar mapeamento final
        mapeamento = {}
        for i in range(1, max_rotulo):
            rotulo = i
            while rotulo in equivalencias:
                rotulo = equivalencias[rotulo]
            mapeamento[i] = rotulo
        
        # Aplicar mapeamento
        rotulos_unicos = np.unique(list(mapeamento.values()))
        mapeamento_final = {old: new for new, old in enumerate(rotulos_unicos, 1)}
        
        resultado = np.zeros_like(rotulada)
        for i in range(rotulada.shape[0]):
            for j in range(rotulada.shape[1]):
                if rotulada[i, j] > 0:
                    resultado[i, j] = mapeamento_final[mapeamento[rotulada[i, j]]]
        
        return resultado, len(rotulos_unicos)
    
    def flood_fill(self, imagem: np.ndarray) -> Tuple[np.ndarray, int]:
        """Implementa rotulagem usando algoritmo flood fill"""
        altura, largura = imagem.shape
        rotulada = np.zeros((altura, largura), dtype=np.int32)
        rotulo_atual = 1
        
        def dfs(i, j, rotulo):
            stack = [(i, j)]
            while stack:
                x, y = stack.pop()
                if 0 <= x < altura and 0 <= y < largura and imagem[x, y] == 255 and rotulada[x, y] == 0:
                    rotulada[x, y] = rotulo
                    # Vizinhos 4-conectados
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        stack.append((x+dx, y+dy))
        
        for i in range(altura):
            for j in range(largura):
                if imagem[i, j] == 255 and rotulada[i, j] == 0:
                    dfs(i, j, rotulo_atual)
                    rotulo_atual += 1
        
        return rotulada, rotulo_atual - 1
    
    def calcular_metricas_componentes(self, imagem_rotulada: np.ndarray, num_componentes: int) -> Dict:
        """Calcula métricas dos componentes conectados"""
        metricas = {
            'num_componentes': num_componentes,
            'areas': [],
            'centroides': [],
            'bboxes': []
        }
        
        for rotulo in range(1, num_componentes + 1):
            mascara = (imagem_rotulada == rotulo)
            area = np.sum(mascara)
            metricas['areas'].append(area)
            
            # Calcular centroide
            y_coords, x_coords = np.where(mascara)
            if len(y_coords) > 0:
                centroide = (np.mean(x_coords), np.mean(y_coords))
                metricas['centroides'].append(centroide)
            
            # Calcular bounding box
            if len(y_coords) > 0:
                bbox = (np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords))
                metricas['bboxes'].append(bbox)
        
        metricas['area_media'] = np.mean(metricas['areas']) if metricas['areas'] else 0
        metricas['area_desvio'] = np.std(metricas['areas']) if metricas['areas'] else 0
        
        return metricas
    
    def gerar_matriz_conectividade(self, imagem_rotulada: np.ndarray) -> np.ndarray:
        """Gera matriz de conectividade entre componentes"""
        num_componentes = np.max(imagem_rotulada)
        matriz = np.zeros((num_componentes, num_componentes), dtype=int)
        
        # Para vizinhança entre componentes (simplificado)
        altura, largura = imagem_rotulada.shape
        for i in range(altura):
            for j in range(largura):
                if imagem_rotulada[i, j] > 0:
                    rotulo_atual = imagem_rotulada[i, j]
                    # Verificar vizinhos
                    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < altura and 0 <= nj < largura:
                            vizinho = imagem_rotulada[ni, nj]
                            if vizinho > 0 and vizinho != rotulo_atual:
                                matriz[rotulo_atual-1, vizinho-1] = 1
        
        return matriz
    
    def visualizar_componentes(self, imagem_rotulada: np.ndarray, titulo: str = ""):
        """Visualiza componentes com cores diferentes"""
        # Gerar cores aleatórias para cada componente
        num_componentes = np.max(imagem_rotulada)
        cores = np.random.randint(0, 255, size=(num_componentes + 1, 3))
        cores[0] = [0, 0, 0]  # Fundo preto
        
        imagem_colorida = np.zeros((*imagem_rotulada.shape, 3), dtype=np.uint8)
        for i in range(imagem_rotulada.shape[0]):
            for j in range(imagem_rotulada.shape[1]):
                rotulo = imagem_rotulada[i, j]
                imagem_colorida[i, j] = cores[rotulo]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(imagem_colorida)
        plt.title(titulo)
        plt.axis('off')
        plt.show()
        
        return imagem_colorida
    
    def comparar_metodos(self, imagem_binaria: np.ndarray) -> pd.DataFrame:
        """Compara diferentes métodos de rotulagem"""
        resultados = []
        
        # Método 1: Vizinhanca-4
        inicio = time.time()
        rotulada_4, num_4 = self.rotulagem_vizinhanca_4(imagem_binaria)
        tempo_4 = time.time() - inicio
        metricas_4 = self.calcular_metricas_componentes(rotulada_4, num_4)
        
        # Método 2: Vizinhanca-8
        inicio = time.time()
        rotulada_8, num_8 = self.rotulagem_vizinhanca_8(imagem_binaria)
        tempo_8 = time.time() - inicio
        metricas_8 = self.calcular_metricas_componentes(rotulada_8, num_8)
        
        # Método 3: Flood Fill
        inicio = time.time()
        rotulada_ff, num_ff = self.flood_fill(imagem_binaria)
        tempo_ff = time.time() - inicio
        metricas_ff = self.calcular_metricas_componentes(rotulada_ff, num_ff)
        
        # Método 4: OpenCV (referência)
        inicio = time.time()
        num_cv, rotulada_cv = cv2.connectedComponents(imagem_binaria)
        tempo_cv = time.time() - inicio
        metricas_cv = self.calcular_metricas_componentes(rotulada_cv, num_cv-1)
        
        resultados.append({
            'Metodo': 'Vizinhanca-4',
            'Num_Componentes': num_4,
            'Tempo(s)': tempo_4,
            'Area_Media': metricas_4['area_media'],
            'Area_Desvio': metricas_4['area_desvio']
        })
        
        resultados.append({
            'Metodo': 'Vizinhanca-8',
            'Num_Componentes': num_8,
            'Tempo(s)': tempo_8,
            'Area_Media': metricas_8['area_media'],
            'Area_Desvio': metricas_8['area_desvio']
        })
        
        resultados.append({
            'Metodo': 'Flood-Fill',
            'Num_Componentes': num_ff,
            'Tempo(s)': tempo_ff,
            'Area_Media': metricas_ff['area_media'],
            'Area_Desvio': metricas_ff['area_desvio']
        })
        
        resultados.append({
            'Metodo': 'OpenCV',
            'Num_Componentes': num_cv-1,
            'Tempo(s)': tempo_cv,
            'Area_Media': metricas_cv['area_media'],
            'Area_Desvio': metricas_cv['area_desvio']
        })
        
        return pd.DataFrame(resultados), rotulada_4, rotulada_8, rotulada_ff, rotulada_cv

# Exemplo de uso e testes
def demonstrar_analise_conectividade():
    # Criar analisador
    analisador = AnalisadorConectividade()
    
    # 1. Criar imagem binária de exemplo
    print("1. Criando imagem binária de exemplo...")
    imagem_teste = np.zeros((100, 100), dtype=np.uint8)
    
    # Adicionar alguns objetos
    cv2.rectangle(imagem_teste, (10, 10), (30, 30), 255, -1)
    cv2.rectangle(imagem_teste, (15, 15), (25, 25), 0, -1)  # Buraco
    cv2.circle(imagem_teste, (70, 30), 15, 255, -1)
    cv2.rectangle(imagem_teste, (50, 60), (80, 80), 255, -1)
    cv2.rectangle(imagem_teste, (55, 65), (75, 75), 255, -1)  # Conectado
    
    # 2. Testar com diferentes limiares (simulado)
    print("2. Testando com diferentes configurações...")
    limiares = [100, 128, 150, 200]
    resultados_limiares = []
    
    for limiar in limiares:
        # Simular diferentes binarizações
        imagem_limiar = analisador.binarizar_imagem(imagem_teste, limiar)
        df, _, _, _, _ = analisador.comparar_metodos(imagem_limiar)
        df['Limiar'] = limiar
        resultados_limiares.append(df)
    
    # 3. Testar com ruído
    print("3. Testando com imagens ruidosas...")
    imagem_ruidosa = analisador.adicionar_ruido(imagem_teste, 'salt', 0.02)
    df_ruido, rot_4_r, rot_8_r, rot_ff_r, rot_cv_r = analisador.comparar_metodos(imagem_ruidosa)
    
    # 4. Comparação principal
    print("4. Comparação principal de métodos...")
    df_comparacao, rot_4, rot_8, rot_ff, rot_cv = analisador.comparar_metodos(imagem_teste)
    
    # 5. Visualizações
    print("5. Gerando visualizações...")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(imagem_teste, cmap='gray')
    plt.title('Imagem Binária Original')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(rot_4, cmap='nipy_spectral')
    plt.title(f'Vizinhanca-4\nComponentes: {np.max(rot_4)}')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(rot_8, cmap='nipy_spectral')
    plt.title(f'Vizinhanca-8\nComponentes: {np.max(rot_8)}')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(rot_ff, cmap='nipy_spectral')
    plt.title(f'Flood-Fill\nComponentes: {np.max(rot_ff)}')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(rot_cv, cmap='nipy_spectral')
    plt.title(f'OpenCV\nComponentes: {np.max(rot_cv)}')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(imagem_ruidosa, cmap='gray')
    plt.title('Imagem com Ruído')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Visualização colorida
    print("6. Visualização colorida dos componentes...")
    analisador.visualizar_componentes(rot_8, "Componentes Conectados - Vizinhanca-8")
    
    # 7. Matriz de conectividade
    print("7. Gerando matriz de conectividade...")
    matriz_conectividade = analisador.gerar_matriz_conectividade(rot_8)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_conectividade, annot=True, cmap='Blues', fmt='d')
    plt.title('Matriz de Conectividade entre Componentes')
    plt.xlabel('Componente')
    plt.ylabel('Componente')
    plt.show()
    
    # 8. Resultados numéricos
    print("8. Resultados comparativos:")
    print("\nComparação entre métodos:")
    print(df_comparacao.to_string(index=False))
    
    print("\nCom ruído:")
    print(df_ruido.to_string(index=False))
    
    # 9. Tabela comparativa com diferentes limiares
    print("\n9. Tabela comparativa com diferentes limiares:")
    df_completo = pd.concat(resultados_limiares, ignore_index=True)
    print(df_completo.pivot(index='Limiar', columns='Metodo', values='Num_Componentes'))
    
    return analisador, df_comparacao, df_ruido

# Executar demonstração
if __name__ == "__main__":
    analisador, resultados, resultados_ruido = demonstrar_analise_conectividade()
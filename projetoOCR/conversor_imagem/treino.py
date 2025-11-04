import os
from tqdm import tqdm
import tensorflow as tf

try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.annotations.images import CVImage
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric


from model import train_model
from configs import ModelConfigs

configuracoes = ModelConfigs()

caminho_dados = "Datasets/90kDICT32px"
caminho_anotacoes_val = caminho_dados + "/annotation_val.txt"
caminho_anotacoes_treino = caminho_dados + "/annotation_train.txt"

# Read metadata file and parse it
def ler_arquivo_anotacao(caminho_anotacao):
    conjunto, vocabulario, max_len = [], set(), 0
    with open(caminho_anotacao, "r") as arquivo:
        for linha in tqdm(arquivo.readlines()):
            linha = linha.split()
            caminho_imagem = caminho_dados + linha[0][1:]
            rotulo = linha[0].split("_")[1]
            conjunto.append([caminho_imagem, rotulo])
            vocabulario.update(list(rotulo))
            max_len = max(max_len, len(rotulo))
    return conjunto, sorted(vocabulario), max_len

conjunto_treino, vocab_treino, max_len_treino = ler_arquivo_anotacao(caminho_anotacoes_treino)
conjunto_val, vocab_val, max_len_val = ler_arquivo_anotacao(caminho_anotacoes_val)

# Save vocab and maximum text length to configs
configuracoes.vocab = "".join(vocab_treino)
configuracoes.max_text_length = max(max_len_treino, max_len_val)
configuracoes.save()

# Create training data provider
provedor_dados_treino = DataProvider(
    dataset=conjunto_treino,
    skip_validation=True,
    batch_size=configuracoes.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configuracoes.width, configuracoes.height),
        LabelIndexer(configuracoes.vocab),
        LabelPadding(max_word_length=configuracoes.max_text_length, padding_value=len(configuracoes.vocab))
        ],
)

# Create validation data provider
provedor_dados_val = DataProvider(
    dataset=conjunto_val,
    skip_validation=True,
    batch_size=configuracoes.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(configuracoes.width, configuracoes.height),
        LabelIndexer(configuracoes.vocab),
        LabelPadding(max_word_length=configuracoes.max_text_length, padding_value=len(configuracoes.vocab))
        ],
)

modelo = train_model(
    input_dim = (configuracoes.height, configuracoes.width, 3),
    output_dim = len(configuracoes.vocab),
)
# Compile the model and print summary
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configuracoes.learning_rate), 
    loss=CTCloss(), 
    metrics=[CWERMetric()],
    run_eagerly=False
)
modelo.summary(line_length=110)

# Define path to save the model
os.makedirs(configuracoes.model_path, exist_ok=True)

# Define callbacks
parada_precoce = EarlyStopping(monitor="val_CER", patience=10, verbose=1)
ponto_verificacao = ModelCheckpoint(f"{configuracoes.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
registrador_treinamento = TrainLogger(configuracoes.model_path)
callback_tensorboard = TensorBoard(f"{configuracoes.model_path}/logs", update_freq=1)
reduzir_lr_por_plat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=5, verbose=1, mode="auto")
modelo_para_onnx = Model2onnx(f"{configuracoes.model_path}/model.h5")

# Train the model
modelo.fit(
    provedor_dados_treino,
    validation_data=provedor_dados_val,
    epochs=configuracoes.train_epochs,
    callbacks=[parada_precoce, ponto_verificacao, registrador_treinamento, reduzir_lr_por_plat, callback_tensorboard, modelo_para_onnx],
    workers=configuracoes.train_workers
)

# Save training and validation datasets as csv files
provedor_dados_treino.to_csv(os.path.join(configuracoes.model_path, "train.csv"))
provedor_dados_val.to_csv(os.path.join(configuracoes.model_path, "val.csv"))
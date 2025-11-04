from keras import camadas
from keras.models import Model

from mltu.tensorflow.model_utils import bloco_residual



def trainamento_modelo(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    entradas  = camadas.Input(shape=input_dim, name="input")
    entrada = camadas.Lambda(lambda x: x/255 )(entradas)

    x1 = bloco_residual(entrada, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    
    x2 = bloco_residual(x1, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x3 = bloco_residual(x2, 16, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    
    x4 = bloco_residual(x3, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x5 = bloco_residual(x4, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x6 = bloco_residual(x5, 64, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x7 = bloco_residual(x6, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    squeezed = camadas.Reshape((x7.shape[-3] * x7.shape[-2], x7.shape[-1]))(x7)

    blstm = camadas.Bidirectional(camadas.LSTM(64, return_sequences=True))(squeezed)

    saidas = camadas.Dense(output_dim + 1, activation="softmax", name="output")(blstm)

    model = Model(inputs=entradas, outputs=saidas)
    return model





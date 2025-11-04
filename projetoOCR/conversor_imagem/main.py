import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class ModeloImagemParaPalavra(OnnxInferenceModel):
    def __init__(self, lista_caracteres: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lista_caracteres = lista_caracteres

    def prever(self, imagem: np.ndarray):
        imagem = cv2.resize(imagem, self.input_shapes[0][1:3][::-1])

        pred_imagem = np.expand_dims(imagem, axis=0).astype(np.float32)

        previsoes = self.model.run(self.output_names, {self.input_names[0]: pred_imagem})[0]

        texto = ctc_decoder(previsoes, self.lista_caracteres)[0]

        return texto


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs
    
    configuracoes = BaseModelConfigs.load("Modelos/conversor_imagem/202211270035/configs.yaml")
    print(configuracoes.model_path)
    modelo = ModeloImagemParaPalavra(model_path=configuracoes.model_path, lista_caracteres=configuracoes.vocab)

    df = pd.read_csv("Modelos/conversor_imagem/202211270035/val.csv").dropna().values.tolist()

    acumulador_cer = []
    for caminho_imagem, rotulo in tqdm(df[:20]):
        imagem = cv2.imread(caminho_imagem.replace("\\", "/"))

        try:
            texto_previsto = modelo.prever(imagem)

            cer = get_cer(texto_previsto, rotulo)
            print(f"Image: {caminho_imagem}, Label: {rotulo}, Prediction: {texto_previsto}, CER: {cer}")

            imagem = cv2.resize(imagem, (imagem.shape[1] * 3, imagem.shape[0] * 3))
            cv2.imshow(texto_previsto, imagem)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            continue
        
        acumulador_cer.append(cer)

    print(f"Average CER: {np.average(acumulador_cer)}")

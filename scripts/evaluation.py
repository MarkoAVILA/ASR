import fire 
from asr_metrics import *
import pandas as pd
import matplotlib.pyplot as plt

def open_file(file):
    preds,refs = [], []
    with open(file,'r') as f:
         for i in f:
            x = i.split('\t')
            preds.append(x[0])
            refs.append(x[1])
    return preds, refs


class EVALUATE:
    def __init__(self, file):
        preds,refs = [], []
        file = file.split(',')
        for f in file:
            p,r = open_file(f)
            preds.append(p)
            refs.append(r)
        self.preds = preds
        self.refs = refs

    def calculate_metrics(self):
        wer, wer_norm, cer, cer_norm = [],[],[],[]
        for p,r in zip(self.preds, self.refs):
            wer.append(WER(p,r))
            wer_norm.append(WER_NORM(p, r))
            cer.append(CER(p, r))
            cer_norm.append(CER_NORM(p, r))
        return wer, wer_norm, cer, cer_norm
    
    def main(self):
        wer, wer_norm, cer, cer_norm = self.calculate_metrics()
        n = len(wer)
        for idx, i in enumerate(range(n)):
            print(f'file {idx}')
            print("Calculating metrics for ASR...")
            print(f"WER ortographique: {wer[i]}")
            print(f"WER normalisé: {wer_norm[i]}")
            print(f"CER: {cer[i]}")
            print(f"CER normalisé: {cer_norm[i]}")

    def graphique(self, list_names):
        wer, wer_norm, cer, cer_norm = self.calculate_metrics()
        df = pd.DataFrame.from_dict({"wer":wer, "wer_norm":wer_norm, "cer":cer, "cer_norm":cer_norm})
        df.index = list(list_names)
        print(df)
        df.to_csv('results.csv', index=True)
        # Graficar un diagrama de barras usando los índices como eje X
        df.plot(kind='bar', legend=True)

        # Añadir etiquetas y título
        plt.xlabel('whisper models')
        plt.ylabel('metrics')
        plt.title('Evaluation du testset PCKT')

        # Guardar el gráfico
        plt.savefig('results.png')

if __name__=='__main__':
    fire.Fire(EVALUATE)
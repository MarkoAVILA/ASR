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

    def graphique(self, list_names, name_graph='results.png', name_df='results.csv'):
        wer, wer_norm, cer, cer_norm = self.calculate_metrics()
        df = pd.DataFrame.from_dict({"wer":wer, "wer_norm":wer_norm, "cer":cer, "cer_norm":cer_norm})
        df.index = list(list_names)
        print(df)
        df.to_csv(name_df, index=True)
        # Graficar un diagrama de barras usando los índices como eje X
        ax = df.plot(kind='bar', legend=True, figsize=(12,12))

        # Añadir etiquetas y título
        plt.xlabel('whisper models', fontsize=14)
        plt.ylabel('Metrics', fontsize=14)
        plt.title('Evaluation du testset PCKT(5039)', fontsize=16)

        # Rotating x-axis labels to avoid overlap (if necessary)
        plt.xticks(rotation=45, ha='right', fontsize=12)

        # Adding a grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        # Añadir los valores encima de cada barra
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=12, padding=3)  # Mostrar los valores con dos decimales

        # Show the plot
        plt.tight_layout()

        # Guardar el gráfico
        plt.savefig(name_graph)

if __name__=='__main__':
    fire.Fire(EVALUATE)
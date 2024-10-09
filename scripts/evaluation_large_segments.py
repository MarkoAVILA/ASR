import fire 
from asr_metrics import *
import pandas as pd
import matplotlib.pyplot as plt

def open_file(file):
    with open(file, 'r') as f:
        l = [i.strip() for i in f]
    return " ".join(l)

class EVALUATION:
    def __init__(self, path_audio, patern='small-ft') -> None:
        with open(path_audio,'r') as f:
            l = [i.strip() for i in f]
        self.refs = [i.split('.wav')[0]+'.tr' for i in l]
        self.patern = patern.split(',')
        l_pred = []
        for p in self.patern:
            l_pred.append([i.split('.wav')[0]+"."+p+'.tr' for i in l])

        self.preds = l_pred
    
    def get_data(self):
        l_ref = [open_file(i) for i in self.refs]
        l_pred =  []
        for pred in self.preds:
            l_pred.append([open_file(i) for i in pred])
        return l_ref, l_pred
    
    def calculate_metrics(self):
        l_ref, l_pred = self.get_data()
        wer, wer_norm, cer, cer_norm = [],[],[],[]
        for idx, pred in enumerate(l_pred):
            print(f"Calculating metrics for ASR in {self.patern[idx]}...")
            print(f"WER ortographique: {WER(pred, l_ref)}")
            print(f"WER normalisé: {WER_NORM(pred, l_ref)}")
            print(f"CER: {CER(pred, l_ref)}")
            print(f"CER normalisé: {CER_NORM(pred, l_ref)}")
            wer.append(WER(pred,l_ref))
            wer_norm.append(WER_NORM(pred, l_ref))
            cer.append(CER(pred, l_ref))
            cer_norm.append(CER_NORM(pred, l_ref))
        return wer, wer_norm, cer, cer_norm

    def graphique(self, name_df, title, name_graph):
        wer, wer_norm, cer, cer_norm = self.calculate_metrics()
        df = pd.DataFrame.from_dict({"wer":wer, "wer_norm":wer_norm, "cer":cer, "cer_norm":cer_norm})
        df.index = self.patern
        print(df)
        df.to_csv(name_df, index=True)
        # Graficar un diagrama de barras usando los índices como eje X
        ax = df.plot(kind='bar', legend=True, figsize=(12,12))

        # Añadir etiquetas y título
        plt.xlabel('whisper models', fontsize=14)
        plt.ylabel('Metrics', fontsize=14)
        plt.title(title, fontsize=16)

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

            

if __name__=="__main__":
    fire.Fire(EVALUATION)


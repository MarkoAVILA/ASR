import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

METRIC_ASR = evaluate.load("wer", max_concurrent_cache_files=20000)
NORMALIZER = BasicTextNormalizer()

def WER(pred_str, label_str):
    # calculer le Wer orthographique
    wer_ortho = 100 * METRIC_ASR.compute(predictions=pred_str, references=label_str)
    
    return wer_ortho

def WER_NORM(pred_str, label_str):
        # calculer le WER normalisé
        pred_str_norm = [NORMALIZER(pred) for pred in pred_str]
        label_str_norm = [NORMALIZER(label) for label in label_str]
        # afin de n'évaluer que les échantillons correspondant à des références non nulles
        pred_str_norm = [
            pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
        ]
        label_str_norm = [
            label_str_norm[i]
            for i in range(len(label_str_norm))
            if len(label_str_norm[i]) > 0
        ]

        wer = 100 * METRIC_ASR.compute(predictions=pred_str_norm, references=label_str_norm)
        return wer

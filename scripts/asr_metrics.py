import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

CER_METRIC = evaluate.load("cer")
WER_METRIC = evaluate.load("wer", max_concurrent_cache_files=20000)
NORMALIZER = BasicTextNormalizer()

def normalization_(pred_str, label_str):
    pred_str = [i.strip() for i in pred_str]
    label_str = [i.strip() for i in label_str]
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
    return pred_str_norm, label_str_norm


def WER(pred_str, label_str):
    # calculer le Wer orthographique
    wer_ortho = 100 * WER_METRIC.compute(predictions=pred_str, references=label_str)
    
    return wer_ortho

def WER_NORM(pred_str, label_str):
        # calculer le WER normalisé
        pred_str_norm, label_str_norm = normalization_(pred_str, label_str)
        wer = 100 * WER_METRIC.compute(predictions=pred_str_norm, references=label_str_norm)
        return wer

def CER(pred_str, label_str):
    cer = 100*CER_METRIC.compute(references=label_str, predictions=pred_str)
    return cer

def CER_NORM(pred_str, label_str):
    pred_norm, label_norm = normalization_(pred_str, label_str)
    cer_norm = 100*CER_METRIC.compute(references=label_norm, predictions=pred_norm)
    return cer_norm

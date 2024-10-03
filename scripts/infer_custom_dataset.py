import fire
import torch
import datasets
from custom_dataset import iter_dataset
from asr_metrics import *
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sacrebleu import corpus_bleu
import logging
logger = logging.getLogger("transformers") # Get the Transformers logger instance
logger.setLevel(logging.INFO)

class CUSTOMINFERENCE:
    def __init__(self, map_data, model_name, data_hf=None, task='transcribe', language='french', skip_special_toks=True, timestamps=False, refs=True):
        self.task = task
        self.reference = refs
        self.skip_special_toks = skip_special_toks
        self.timestamps = timestamps
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        logging.info(f'Device is {self.device}')
        logging.info(f'Loading {model_name}... model')
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        logging.info(f'Model loaded!')
        self.model.config.use_cache = True
        self.model.to(self.device)
        logging.info(f'Loading Processor whisper ...')
        self.processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task, predict_timestamps=timestamps)
        logging.info(f'Porcessor loaded!')
        if self.task!='transcribe':
            self.model.config.forced_decoder_ids = None ### initially no token is forced in generation (context tokens)
        self.model.generation_config.task = "transcribe"

        logging.info(f'Loading Dataset...')
        if data_hf is None:
            self.ds = iter_dataset(map_data)
            cpt = 0
            for _ in self.ds:
                cpt+=1
            logger.info('Dataset with {} elements'.format(cpt))
        else:
            self.ds = data_hf
            logger.info('Dataset with {} elements'.format(len(self.ds)))

    def inference(self,output):
        f_out = open(output,'w')
        preds = []
        refs = []
        for el in self.ds:
            input_features = self.processor(el["audio"]['array'], sampling_rate=el["audio"]["sampling_rate"], return_tensors="pt").input_features
            input_features = input_features.to(self.device)
            predicted_ids = self.model.generate(input_features, task="transcribe", language=el["tgt_lang"], return_timestamps=self.timestamps)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=self.skip_special_toks)[0].strip()
            preds.append(transcription)
            if self.task=='transcribe':
                refs.append(el['transcription'])
            else:
                refs.append(el['translation'])
            f_out.write(preds[-1] + ('\t'+refs[-1] if self.reference else '') + "\n")
            f_out.flush()
        print("Calculating metrics for ASR...")
        print(f"WER ortographique: {WER(preds, refs)}")
        print(f"WER normalisé: {WER_NORM(preds, refs)}")
        print(f"CER: {CER(preds, refs)}")
        print(f"CER normalisé: {CER_NORM(preds, refs)}")
        if self.task!='transcribe':
            bleu = round(corpus_bleu(preds, [refs]).score,3)
            print(f"BLEU: {bleu}")

        
if __name__=="__main__":
    fire.Fire(CUSTOMINFERENCE)
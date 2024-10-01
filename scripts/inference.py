import sys
import fire
import torch
import evaluate
import argparse
import datasets
from datasets import load_dataset, Audio, DatasetDict
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
sys.path.append("/nfs/RESEARCH/avila/ASR/")
from sacrebleu import corpus_bleu
from custom_dataset import get_test_dataset
import logging
logger = logging.getLogger("transformers") # Get the Transformers logger instance
logger.setLevel(logging.INFO)



def file2list(path):
    with open(path, 'r') as f:
        l = [i.strip() for i in f]
    return l

def main(data_dir,out, model_base=None, language="catalan",data_id="covost2", code_lang="en_ca", skip_special_toks=True,timestamps=False, data_ref="translation", output_reference=True):
    # CUDA or not CUDA
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f'device is {device}')
    # torch_dtype = torch.float32 #torch.float16 if torch.cuda.is_available() else torch.float32
    # dataset_voice = load_dataset(data_id, code_lang, data_dir=data_dir)

    # Load model and processor
    processor = WhisperProcessor.from_pretrained(model_base, language=language, task='transcribe', predict_timestamps=timestamps)
    model= WhisperForConditionalGeneration.from_pretrained(model_base)
    model.config.use_cache = False
    #model.config.suppress_tokens = []
    model.to(device)

    # indicate context tokens for generation (the next 4 lines are not needed if language,task,timestamps are passed in model.generate)
    model.config.forced_decoder_ids = None ### initially no token is forced in generation (context tokens)
    # model.generation_config.language = language
    model.generation_config.task = "transcribe"
    # model.generation_config.return_timestamps = args.timestamps
    
   
    # load dataset and read audio files
    # ds = DatasetDict()
    ds = datasets.load_from_disk('test.hf') #data/val.hf
    logger.info('loading dataset')
    # ds = load_dataset(data_id, code_lang,split="validation", data_dir=data_dir) #split=args"test[:5]"
    # ds = get_test_dataset(data_dir, code_lang)
    logger.info('dataset with {} elements'.format(len(ds)))
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    logger.info('resampled to 16k')
    # ds = ds.select(range(0,1000))
    # print(ds)
    # ds = ds.select(range(len(ds)//2,len(ds)))
    # print(ds)
    print(ds)

    fout=open(out, 'w')
    fwer=open(out+".wer", 'w')
    fbleu=open(out+".bleu", 'w')
    
    refs = []
    preds = []
    for i in range(len(ds)):
        input_features = processor(ds[i]["audio"]['array'], sampling_rate=ds[i]["audio"]["sampling_rate"], return_tensors="pt").input_features
        input_features = input_features.to(device)
        predicted_ids = model.generate(input_features, task='transcribe', language=ds[i]["tgt_lang"], return_timestamps=timestamps)
        # predicted_ids = model.generate(input_features, task='transcribe', language=language, return_timestamps=timestamps)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=skip_special_toks)[0].strip()
        preds.append(transcription)
        if data_ref in ds[i]:
            refs.append(ds[i][data_ref])

        fout.write(transcription + ('\t'+ds[i][data_ref] if output_reference and data_ref in ds[i] else '') + "\n")
        fout.flush()
        logger.info(transcription)

    if len(refs):
        metric = evaluate.load("wer")
        wer = 100 * metric.compute(predictions=preds, references=refs)
        fwer.write('wer: {:.2f}\n'.format(wer))
        bleu = round(corpus_bleu(preds, [refs]).score,3)
        fbleu.write('bleu:{:.3f}\n'.format(bleu))

        
if __name__=="__main__":
    fire.Fire(main)
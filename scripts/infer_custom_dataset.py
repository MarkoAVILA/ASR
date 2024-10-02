import fire
import torch
import evaluate
import datasets
from custom_dataset import iter_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sacrebleu import corpus_bleu
import logging
logger = logging.getLogger("transformers") # Get the Transformers logger instance
logger.setLevel(logging.INFO)


def main(model,map_data, output, language="catalan", skip_special_toks=True,timestamps=False, output_reference=True):
    # CUDA or not CUDA
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f'device is {device}')
    # torch_dtype = torch.float32 #torch.float16 if torch.cuda.is_available() else torch.float32
    # dataset_voice = load_dataset(data_id, code_lang, data_dir=data_dir)

    # Load model and processor
    processor = WhisperProcessor.from_pretrained(model, language=language, task='transcribe', predict_timestamps=timestamps)
    model= WhisperForConditionalGeneration.from_pretrained(model)
    model.config.use_cache = True
    model.to(device)

    # indicate context tokens for generation (the next 4 lines are not needed if language,task,timestamps are passed in model.generate)
    # model.config.forced_decoder_ids = None ### initially no token is forced in generation (context tokens)
    model.generation_config.task = "transcribe"
    
   
    # load dataset and read audio files
    logger.info('Loading Dataset upsampling to 16KHz')
    ds = iter_dataset(map_data)
    logger.info('Dataset with {} elements'.format(len(ds)))
    logger.info('resampled to 16k')
    print(ds)

    fout=open(output, 'w')
    fwer=open(output+".wer", 'w')

    refs = []
    preds = []
    for i in range(len(ds)):
        input_features = processor(ds[i]["audio"]['array'], sampling_rate=ds[i]["audio"]["sampling_rate"], return_tensors="pt").input_features
        input_features = input_features.to(device)
        predicted_ids = model.generate(input_features, task='transcribe', language=ds[i]["tgt_lang"], return_timestamps=timestamps)
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
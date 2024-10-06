import fire
from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE
from transformers import WhisperProcessor, WhisperTokenizer
from transformers import WhisperForConditionalGeneration
from custom_dataset import iter_dataset
import datasets
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from wer_metric import WER, WER_NORM
from sacrebleu import corpus_bleu



@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Diviser les entrées et les étiquettes car elles doivent être de longueurs différentes et nécessitent des méthodes de remplissage différentes
        # traiter d'abord les entrées audio en renvoyant simplement des tenseurs Torch
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        print("BATCH")
        print(batch["input_features"].shape)
        # obtenir les séquences d'étiquettes tokenisées
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # rembourrer les étiquettes à la longueur maximale
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # remplacer le remplissage par -100 pour ignorer correctement les pertes
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # si le token bos est ajouté lors de l'étape de tokenisation précédente, couper le token bos ici puisqu'il sera de toute façon ajouté plus tard
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        print(batch["labels"].shape)

        return batch
    

class WHISPER_FT:
    def __init__(self, path_model, data_train, data_val, num_proc, langue_1="catalan", type_task="Speech2Text") -> None:

        self.num_proc = num_proc
        self.type_task = type_task
        self.langue_1 = langue_1
        self.path_model = path_model
        self.data_train = data_train
        self.data_val = data_val


        if torch.cuda.is_available():
            print("GPU is avalaible with {} device(s)".format(torch.cuda.device_count()), flush=True)
        else:
            print("GPU is not avalaible", flush=True)


        self.model = WhisperForConditionalGeneration.from_pretrained(self.path_model)
    
        self.processor = WhisperProcessor.from_pretrained(self.path_model, language=self.langue_1, task="transcribe")
        self.processor_1 = WhisperProcessor.from_pretrained(self.path_model, language=self.langue_1, task="transcribe")
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)

        # désactiver le cache pendant l'entraînement car il est incompatible avec le checkpointing du gradient
        self.model.config.use_cache = False
        self.model.generation_config.language = 'french'
        self.model.generation_config.task = 'transcribe'
        self.model.config.forced_decoder_ids = None #Remove forced_decoder_ids when using task="transcribe" to avoid conflicts.
        self.model.config.suppress_tokens = []

    def change_type_audio(self, example):
        example['audio']['array'] = np.array(example['audio']['array'], dtype='float32')
        return example
    
    def loading_dataset(self):
        if '.hf' in self.data_train or '.hf' in self.data_val:
            dataset_train, dataset_validation= datasets.load_from_disk(self.data_train), datasets.load_from_disk(self.data_val)
            print("nb samples train:", len(dataset_train), flush=True)
            print("nb samples valid:", len(dataset_validation),flush=True)
        else:
            dataset_train, dataset_validation = iter_dataset(self.data_train), iter_dataset(self.data_val, type='test')
            print("example de train", next(iter(dataset_train)), flush=True)
            print("example de validation", next(iter(dataset_validation)), flush=True)

        print("dataset loaded!", flush=True)
        return dataset_train, dataset_validation
    
    
    def feature_extractor(self, example):
        # self.processor.tokenizer.set_prefix_tokens(language=example["tgt_lang"], task='transcribe')
        example_audio, sentence = example['audio'], example["transcription"]
        audio, sr = np.array(example_audio['array'], dtype='float32'), example_audio["sampling_rate"]
        example = self.processor(
            audio = audio,
            sampling_rate = sr,
            text = sentence
        )
        # compute input length of audio sample in seconds
        example["input_length"] = len(audio) / sr

        return example

    def selection(self, tgt_lang):
        return tgt_lang==self.langue_1
    
    def building_dataset_voice(self):
        dataset_train, dataset_validation = self.loading_dataset()
        dataset_validation = dataset_validation.filter(self.selection, input_columns=["tgt_lang"])
        dataset_voice_train = dataset_train.map(self.feature_extractor)
        dataset_voice_test = dataset_validation.map(self.feature_extractor)
        print("Mapping features extractor done!", flush=True)
        return dataset_voice_train, dataset_voice_test

    def compute_metrics(self, pred):
        pred_ids, label_ids  = pred.predictions, pred.label_ids
        # remplacer -100 par pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        # nous ne voulons pas grouper les *tokens* lors du calcul des métriques
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        # print("PRED")
        # print(pred_str)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)
        # print("LABELS")
        # print(label_str)
        wer_ortho, wer = WER(pred_str, label_str), WER_NORM(pred_str, label_str)
        bleu = corpus_bleu(pred_str, [label_str])
        return {"bleu":round(bleu.score,3), "wer_ortho": wer_ortho, "wer": wer}


    def train(self, output_dir="./whisper-small-dv",
            #   num_train_epochs=1,
              per_device_train_batch_size=8,#16
              gradient_accumulation_steps=1,
              learning_rate=1e-5, lr_scheduler_type="constant_with_warmup",
              warmup_steps=50,
              max_steps=500,
              gradient_checkpointing=True,
              fp16=True,
              fp16_full_eval=True,
              evaluation_strategy="steps",
              per_device_eval_batch_size=16,
              predict_with_generate=True,
              generation_max_length=225,
              save_steps=5000,
              eval_steps=500,
              logging_steps=25,
              report_to=["tensorboard"],
              load_best_model_at_end=True,
              metric_for_best_model="wer",
              greater_is_better=False,
              push_to_hub=True,
              ):
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,  # nom sur le Hub,
            # num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,  # à x2  pour chaque diminution de 2x de la taille du batch
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            max_steps=max_steps,  # augmenter jusqu'à 4000 si vous disposez de votre propre GPU ou d'un plan Colab payant
            gradient_checkpointing=gradient_checkpointing,
            fp16=fp16,
            fp16_full_eval=fp16_full_eval,
            evaluation_strategy=evaluation_strategy,
            per_device_eval_batch_size=per_device_eval_batch_size,
            predict_with_generate=predict_with_generate,
            generation_max_length=generation_max_length,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            report_to=report_to,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            push_to_hub=push_to_hub,
            remove_unused_columns=False
            )
        print(learning_rate)
        print(type(learning_rate))
        
        dataset_train, dataset_test = self.building_dataset_voice()

        self.processor.tokenizer.set_prefix_tokens(language=self.langue_1, task='transcribe') #language may be None
        trainer = Seq2SeqTrainer(
                args=training_args,
                model=self.model,
                train_dataset=dataset_train,
                eval_dataset=dataset_test,
                data_collator=self.data_collator,
                compute_metrics=self.compute_metrics,
                tokenizer=self.processor,
                )

        trainer.train()


if __name__=="__main__":
    fire.Fire(WHISPER_FT)
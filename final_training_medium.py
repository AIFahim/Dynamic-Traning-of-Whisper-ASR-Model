## Setting Up Environment Variables & Devices
import os
import pandas as pd
import traceback
from statistics import mode
# import comm
import torch
from train_val_df_gen import Train_Val_df
from datasets import Dataset, DatasetDict, Audio
# import mlflow
# import mlflow.pytorch
import pydub
import json
import librosa
import itertools
import soundfile as sf 

import math
from audioaugmentations import AudioAugmentations
import random
import traceback
import torchaudio
import numpy as np
import glob
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperTokenizerFast, WhisperProcessor
from datasets import load_dataset, DatasetDict
import evaluate
from dataclasses import dataclass
from typing import Any, Callable , Dict, List, Union
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from bnunicodenormalizer import Normalizer
import unicodedata
import re
import pickle
import time
import redis
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer , Trainer
import transformers as tf

redisClient = redis.Redis(host="localhost", port=6380)


## Setting Up Training Args
# NUM_PROC = 1
# NUM_CHUNKS = 1
# NUM_EPOCHS = 1
# TOTAL_FILES = 8
os.environ['MLFLOW_TRACKING_USERNAME'] = "mlflow"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "1234567"
abs_path = os.path.abspath('.')
base_dir = os.path.dirname(abs_path)
device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"\n\n Device to be used: {device} \n\n")
model_name = "openai/whisper-medium"
language = "Bengali"
task = "transcribe"
output_dir = "./"
overwrite_output_dir = True
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
# gradient_accumulation_steps = 1
# max_steps =  #math.ceil((TOTAL_FILES // NUM_CHUNKS) / (per_device_train_batch_size )) * NUM_EPOCHS  # 3000
max_steps = 3475830 # 434478 #(5 epochs) # 871333 #math.ceil((TOTAL_FILES // NUM_CHUNKS) / (per_device_train_batch_size )) * NUM_EPOCHS
# dataloader_num_workers = 1
gradient_checkpointing = False
evaluation_strategy = "steps"
eval_steps = 2000
save_strategy = "steps"
save_steps = 2000
# save_steps = 5
save_total_limit = 2
learning_rate = 1e-5
lr_scheduler_type = "cosine_with_restarts"  # "cosine" # "constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear"(default), "polynomial", "inverse_sqrt"
warmup_steps = 888  # (1 epoch)
logging_steps = 1  # 25
weight_decay = 0
dropout = 0  # 0.1  # any value > 0.1 hurts performance. So, use values between 0.0 and 0.1
load_best_model_at_end = True
metric_for_best_model = "cer"
greater_is_better = False
bf16 = False
tf32 = True
generation_max_length = 448
predict_with_generate = True
push_to_hub = True
freeze_feature_encoder = False
early_stopping_patience = 10
apply_spec_augment = False

## Load Datasets
print("\n\n Loading Datasets...this might take a while..\n\n")
with open('/home/asif/merged_dict.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Call the generate_df_from_json() method on the Train_Val_df class directly
train_df, val_df, split_index = Train_Val_df.generate_df_from_json(json_data)
print("Total Datapoint Lenghts", len(train_df), len(val_df))

# Define the number of dummy data points
num_data_points = split_index * 2
data_list = [('path{}.flac'.format(i), 'text{}'.format(i)) for i in range(1, num_data_points+1)]
# Unpack the list into two separate lists
flac_path, txt_list = zip(*data_list)
# Create a dictionary with the data
data = {
    'input_features': flac_path,
    'labels': txt_list,
}

# Create a pandas DataFrame
dummy_train_df = pd.DataFrame(data)
dataset_our = DatasetDict({
    "train": Dataset.from_pandas(dummy_train_df),
    "test": Dataset.from_pandas(val_df)
})


## Prepare Feature Extractor, Tokenizer and Processor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-medium", language=language, task=task)
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language=language, task=task)

## Define Data Collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    forward_attention_mask: bool
    def __call__(self, dataset: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        
        if "path" in dataset[0]["input_features"]:
            keys = redisClient.keys()
            while(True):
                if len(keys) >= per_device_train_batch_size:
                    break
                print("sleeping started ......")
                print(" len(keys) ",len(keys))
                time.sleep(300)
                keys = redisClient.keys()
            dataset_prepared = []
            for key in redisClient.keys()[:per_device_train_batch_size]:
                redis_bytes = redisClient.get(key)
                try:
                    redis_array = pickle.loads(redis_bytes)
                    dataset_prepared.append({
                        "input_features": redis_array[0],
                        "labels" : processor.tokenizer(redis_array[1]).input_ids,
                    })
                except:
                    redisClient.delete(key)
            # After applying augmentations, continue with the original data collator code
            input_features = [{"input_features": feature["input_features"]} for feature in dataset_prepared]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            if self.forward_attention_mask:
                batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in dataset_prepared])
            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in dataset_prepared]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch
        
        else:
            dataset_prepared = []
            for sample in dataset:
                load_ori, sr = torchaudio.load(sample["input_features"])
                load_ori = load_ori.to(torch.float64).numpy()
                if load_ori.ndim == 2 and load_ori.shape[0] == 1:
                    load_ori = np.squeeze(load_ori, axis=(0))
                norm = librosa.util.normalize(load_ori)
                norm  = librosa.to_mono(norm)
                inputs = processor.feature_extractor(
                    norm,
                    sampling_rate=16000,
                    return_attention_mask=False,
                )
                input_features = inputs.input_features[0]
                labels = processor.tokenizer(sample["labels"]).input_ids
                dataset_prepared.append({
                    "input_features": input_features,
                    "labels" : labels,
                })
            
            # After applying augmentations, continue with the original data collator code
            input_features = [{"input_features": feature["input_features"]} for feature in dataset_prepared]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            if self.forward_attention_mask:
                batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in dataset_prepared])

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in dataset_prepared]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            
            return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    forward_attention_mask=apply_spec_augment,
)


## Define Evaluation Metrics
wer_metric = evaluate.load("wer", cache_dir=os.path.join(base_dir, "metrics_cache"))
cer_metric = evaluate.load("cer", cache_dir=os.path.join(base_dir, "metrics_cache"))
do_normalize_eval = True
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}



## Load a Pre-Trained Checkpoint
print("\n\n Loading Model to Device..\n\n")
model = WhisperForConditionalGeneration.from_pretrained("/home/asif/whisper_900_hr_dataset/scripts/DyT_PoCs_2/900hr_augmented_whisper_medium_5th_try/checkpoint-188000")
## Override generation arguments
model.config.apply_spec_augment = apply_spec_augment
model.config.dropout = dropout
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
if gradient_checkpointing:
    model.config.use_cache = False
if freeze_feature_encoder:
    model.freeze_feature_encoder()

## Define the Training Configuration
training_args = Seq2SeqTrainingArguments(
    output_dir="900hr_augmented_whisper_medium_5th_try_after_188000_checkpoint",
    # overwrite_output_dir=overwrite_output_dir,
    max_steps=max_steps,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_checkpointing=gradient_checkpointing,
    evaluation_strategy=evaluation_strategy,
    eval_steps=eval_steps,
    save_strategy=save_strategy,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    warmup_steps=warmup_steps,
    logging_steps=logging_steps,
    weight_decay=weight_decay,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=metric_for_best_model,
    greater_is_better=greater_is_better,
    bf16=bf16,
    tf32=tf32,
    generation_max_length=generation_max_length,
    predict_with_generate=predict_with_generate,
    push_to_hub=push_to_hub,
    hub_token="my_hub_token",
)



trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset_our["train"],
    eval_dataset=dataset_our["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained("best_model")

## Training
print("\n\n Training STARTED..\n\n")

train_result = trainer.train()


import os
import traceback
from statistics import mode
import torch
from train_val_df_gen import Train_Val_df
from datasets import Dataset, DatasetDict, Audio
import json
import librosa
import soundfile as sf
from audioaugmentations_w_speech_enhance import AudioAugmentations
import random
import traceback
import torchaudio
import numpy as np
import glob
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperTokenizerFast, WhisperProcessor
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Union
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from bnunicodenormalizer import Normalizer
import redis
import pickle
import uuid
import time
import redis
from speechbrain.pretrained import SepformerSeparation as separator

redisClient = redis.Redis(host="localhost", port=6380)
REDISSIZE =  49
# ENHANCED_MODEL = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='pretrained_models/sepformer-wham16k-enhancement',run_opts={"device":"cpu"})

model_name = "openai/whisper-small"
language = "Bengali"
task = "transcribe"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-medium", language=language, task=task)
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language=language, task=task)

def prepare_dataset2(sample):
    try:
        norm = librosa.util.normalize(sample)
        norm  = librosa.to_mono(norm)
        inputs = processor.feature_extractor(
            norm,
            sampling_rate=16000,
            return_attention_mask=False,
        )
        # print(torch.from_numpy(inputs.input_features[0]).size())
        return inputs.input_features[0]

    except Exception as e:
        print(f"Skipping this batch due to an exception: {e}")
        return None


bgNoiseFileList = glob.glob("/home/asif/augmentations_experiments/environmental-sound-classification-50/audio/audio/16000/**/*.wav", recursive=True)

with open('/home/asif/merged_dict.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

train_df, val_df, split_index = Train_Val_df.generate_df_from_json(json_data)
# train_df = train_df[:40]


APPLIED_AUGMENTATIONS = [
    "speedAug",
    "pitchShift",
    "farFieldEffect",
    "colorNoise",
    "bgNoiseAug",
    "down_upsampling",
    "time_n_freq_masking",
]

epoch_cnt = 1

while True:
    sample_cnt = 0
    option_idx = [0] * len(APPLIED_AUGMENTATIONS)

    cnt = 0
    for idx, sample in train_df.iterrows():

        if cnt == len(APPLIED_AUGMENTATIONS):
            cnt = 0

        AudioAugmentations_pera = [0.0] * len(APPLIED_AUGMENTATIONS)
        AudioAugmentations_pera[cnt] += 1.0

        Choices = {
            str(APPLIED_AUGMENTATIONS.index("speedAug")): [0.75, 0.8, 0.9, 1.1, 1.25, 1.5],  # Speed
            str(APPLIED_AUGMENTATIONS.index("pitchShift")): [3, -3],  # Pitch
            str(APPLIED_AUGMENTATIONS.index("farFieldEffect")): [1.0, 3.0, 5.0],  # Far Field
            str(APPLIED_AUGMENTATIONS.index("bgNoiseAug")): [0.8, 0.9, 0.95],  # BG Focus Blur
            str(APPLIED_AUGMENTATIONS.index("down_upsampling")): [2000, 4000, 8000]
        }
        # if cnt != 0:
        if str(cnt) in Choices:
            option = Choices[str(cnt)][option_idx[cnt]]
            audio_augmentor = AudioAugmentations(
                multipleAug=False,
                speedAugProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("speedAug")],
                pitchShiftProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("pitchShift")],
                farFieldEffectProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("farFieldEffect")],
                bgNoiseAugProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("bgNoiseAug")],
                colorNoiseProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("colorNoise")],
                time_n_freq_maskingProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("time_n_freq_masking")],
                down_upsamplingProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("down_upsampling")],
                bgNoise_focusMinMax=[option],
                speedFactors=[option],
                pitchShiftRange=[option],
                farFieldDistances=[option],
                bgNoiseFileList=bgNoiseFileList,
                down_upsamplingMinMax=[option],
                sampleRate=16000,
                # enhanced_model = ENHANCED_MODEL
            )
            try:
                augmented = audio_augmentor.getAudio(sample["input_features"], returnTensor=False)
                processed_augmented = prepare_dataset2(augmented)
                if processed_augmented is not None:
                    processed_augmented = (processed_augmented, sample["labels"])
                    processed_augmented = pickle.dumps(processed_augmented)


                
                    while True:
                        # if redisClient.dbsize() <= REDISSIZE:
                        if (redisClient.info('memory')['used_memory']/1073741824) <= REDISSIZE:
                            key = str(uuid.uuid4())
                            redisClient.set(key, processed_augmented)
                            sample_cnt+=1
                            break
                        # time.sleep(5)
            except:
                import traceback
                with open("/home/asif/whisper_900_hr_dataset/scripts/DyT_PoCs_2/exceptions.txt", "a") as f:
                    f.write(traceback.format_exc()+"\n")
                pass

            option_idx[cnt] = (option_idx[cnt] + 1) % len(Choices[str(cnt)])
        else:
            audio_augmentor = AudioAugmentations(
                multipleAug=False,
                speedAugProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("speedAug")],
                pitchShiftProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("pitchShift")],
                farFieldEffectProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("farFieldEffect")],
                bgNoiseAugProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("bgNoiseAug")],
                colorNoiseProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("colorNoise")],
                time_n_freq_maskingProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("time_n_freq_masking")],
                down_upsamplingProb=AudioAugmentations_pera[APPLIED_AUGMENTATIONS.index("down_upsampling")],
                bgNoiseFileList=bgNoiseFileList,
                # enhanced_model = ENHANCED_MODEL,

                sampleRate=16000,
            )
            
            try:
                augmented = audio_augmentor.getAudio(sample["input_features"], returnTensor=False)
                processed_augmented = prepare_dataset2(augmented)

                if processed_augmented is not None:
                    processed_augmented = (processed_augmented, sample["labels"])
                    processed_augmented = pickle.dumps(processed_augmented)


                    while True:
                        # if redisClient.dbsize() <= REDISSIZE:
                        if (redisClient.info('memory')['used_memory']/1073741824) <= REDISSIZE:
                            key = str(uuid.uuid4())
                            redisClient.set(key, processed_augmented)
                            sample_cnt+=1
                            break
                    # time.sleep(5)
            except:
                import traceback
                with open("/home/asif/whisper_900_hr_dataset/scripts/DyT_PoCs_2/exceptions.txt", "a") as f:
                    f.write(traceback.format_exc()+"\n")
                pass

        # load_ori, sr = torchaudio.load(sample["input_features"])
        # load_ori = load_ori.to(torch.float64).numpy()
        
        load_ori , sr = librosa.load(path=sample["input_features"],sr=None,mono=True)
        if load_ori.ndim == 2 and load_ori.shape[0] == 1:
            load_ori = np.squeeze(load_ori, axis=(0))
        
        processed_original = prepare_dataset2(load_ori)

        if processed_original is not None:

            processed_original = (processed_original, sample["labels"])
            processed_original = pickle.dumps(processed_original)



            while True:
                # if redisClient.dbsize() <= REDISSIZE:
                if (redisClient.info('memory')['used_memory']/1073741824) <= REDISSIZE:
                    key = str(uuid.uuid4())
                    redisClient.set(key, processed_original)
                    sample_cnt+=1
                    break
                # time.sleep(5)
        
        if sample_cnt%10000<=10 and sample_cnt>=10000:
            with open("preprocess_logs.txt", "a") as f:
                f.write(f"{sample_cnt} Samples Completed\n")



        cnt += 1
    with open("preprocess_logs.txt", "a") as f:
        f.write(f"{epoch_cnt} Epochs Completed!!!!!!!!!!!!!!!!!!!!\n")
    epoch_cnt+=1








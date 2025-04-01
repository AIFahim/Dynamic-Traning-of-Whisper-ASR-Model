import torch
import torchaudio
import numpy as np
import librosa
from typing import Union, Dict, Any
from transformers import WhisperProcessor
from config import AudioAugmentationConfig

class DataProcessor:
    def __init__(self, processor: WhisperProcessor, config: AudioAugmentationConfig):
        self.processor = processor
        self.config = config

    def prepare_dataset(self, audio_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Prepare audio data for the model."""
        try:
            # Convert to numpy if tensor
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.numpy()

            # Normalize and convert to mono
            norm = librosa.util.normalize(audio_data)
            norm = librosa.to_mono(norm)

            # Extract features
            inputs = self.processor.feature_extractor(
                norm,
                sampling_rate=self.config.sample_rate,
                return_attention_mask=False,
            )
            return inputs.input_features[0]
        except Exception as e:
            print(f"Error preparing dataset: {e}")
            return None

    def process_audio_file(self, file_path: str) -> np.ndarray:
        """Process an audio file and return its features."""
        try:
            # Load audio file
            audio_data, sr = librosa.load(path=file_path, sr=None, mono=True)
            
            # Ensure correct shape
            if audio_data.ndim == 2 and audio_data.shape[0] == 1:
                audio_data = np.squeeze(audio_data, axis=(0))

            # Prepare features
            return self.prepare_dataset(audio_data)
        except Exception as e:
            print(f"Error processing audio file {file_path}: {e}")
            return None

    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare a batch of data for training."""
        try:
            # Prepare input features
            input_features = [{"input_features": feature["input_features"]} for feature in batch]
            processed_batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # Add attention mask if needed
            if self.config.apply_spec_augment:
                processed_batch["attention_mask"] = torch.LongTensor(
                    [feature["attention_mask"] for feature in batch]
                )

            # Process labels
            label_features = [{"input_ids": feature["labels"]} for feature in batch]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            
            # Replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # Remove BOS token if present
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            processed_batch["labels"] = labels
            return processed_batch
        except Exception as e:
            print(f"Error preparing batch: {e}")
            return None 
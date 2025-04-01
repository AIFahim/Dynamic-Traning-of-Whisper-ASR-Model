from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6380
    max_memory_gb: float = 49.0

@dataclass
class ModelConfig:
    model_name: str = "openai/whisper-medium"
    language: str = "Bengali"
    task: str = "transcribe"
    output_dir: str = "./"
    overwrite_output_dir: bool = True
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    max_steps: int = 3475830
    gradient_checkpointing: bool = False
    evaluation_strategy: str = "steps"
    eval_steps: int = 2000
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine_with_restarts"
    warmup_steps: int = 888
    logging_steps: int = 1
    weight_decay: float = 0
    dropout: float = 0
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "cer"
    greater_is_better: bool = False
    bf16: bool = False
    tf32: bool = True
    generation_max_length: int = 448
    predict_with_generate: bool = True
    push_to_hub: bool = True
    freeze_feature_encoder: bool = False
    early_stopping_patience: int = 10
    apply_spec_augment: bool = False

@dataclass
class AudioAugmentationConfig:
    sample_rate: int = 16000
    multiple_aug: bool = False
    speed_aug_prob: float = 0.5
    pitch_shift_prob: float = 0.25
    far_field_effect_prob: float = 0.25
    bg_noise_aug_prob: float = 0.25
    color_noise_prob: float = 0.25
    time_n_freq_masking_prob: float = 0.25
    down_upsampling_prob: float = 0.25
    speech_enhance_prob: float = 0.25
    
    speed_factors: List[float] = None
    pitch_shift_range: List[int] = None
    far_field_distances: List[float] = None
    bg_noise_focus_min_max: List[float] = None
    noise_gain_min_max: List[float] = None
    down_upsampling_min_max: List[int] = None
    bg_noise_file_list: List[str] = None

    def __post_init__(self):
        if self.speed_factors is None:
            self.speed_factors = [0.75, 0.8, 0.9, 1.1, 1.25, 1.5]
        if self.pitch_shift_range is None:
            self.pitch_shift_range = [3, -3]
        if self.far_field_distances is None:
            self.far_field_distances = [1.0, 3.0, 5.0]
        if self.bg_noise_focus_min_max is None:
            self.bg_noise_focus_min_max = [0.8, 0.9, 0.95]
        if self.noise_gain_min_max is None:
            self.noise_gain_min_max = [0.1]
        if self.down_upsampling_min_max is None:
            self.down_upsampling_min_max = [2000, 4000, 8000]

@dataclass
class Config:
    redis: RedisConfig = RedisConfig()
    model: ModelConfig = ModelConfig()
    audio_augmentation: AudioAugmentationConfig = AudioAugmentationConfig() 
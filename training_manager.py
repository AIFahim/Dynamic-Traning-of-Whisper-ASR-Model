import os
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset, DatasetDict
import evaluate
from typing import Dict, Any
from config import ModelConfig
from data_processor import DataProcessor
from redis_manager import RedisManager

class TrainingManager:
    def __init__(
        self,
        model_config: ModelConfig,
        data_processor: DataProcessor,
        redis_manager: RedisManager
    ):
        self.config = model_config
        self.data_processor = data_processor
        self.redis_manager = redis_manager
        self.device = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"\n\n Device to be used: {self.device} \n\n")

        # Initialize model components
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_config.model_name)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(
            model_config.model_name,
            language=model_config.language,
            task=model_config.task
        )
        self.processor = WhisperProcessor.from_pretrained(
            model_config.model_name,
            language=model_config.language,
            task=model_config.task
        )
        self.model = self._load_model()

    def _load_model(self) -> WhisperForConditionalGeneration:
        """Load and configure the model."""
        model = WhisperForConditionalGeneration.from_pretrained(self.config.model_name)
        
        # Configure model
        model.config.apply_spec_augment = self.config.apply_spec_augment
        model.config.dropout = self.config.dropout
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        
        if self.config.gradient_checkpointing:
            model.config.use_cache = False
            
        if self.config.freeze_feature_encoder:
            model.freeze_feature_encoder()
            
        return model

    def _create_training_args(self) -> Seq2SeqTrainingArguments:
        """Create training arguments."""
        return Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            max_steps=self.config.max_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_checkpointing=self.config.gradient_checkpointing,
            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            weight_decay=self.config.weight_decay,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            bf16=self.config.bf16,
            tf32=self.config.tf32,
            generation_max_length=self.config.generation_max_length,
            predict_with_generate=self.config.predict_with_generate,
            push_to_hub=self.config.push_to_hub,
            hub_token="my_hub_token",
        )

    def _compute_metrics(self, pred):
        """Compute evaluation metrics."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # Decode predictions and labels
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute metrics
        wer = evaluate.load("wer").compute(predictions=pred_str, references=label_str)
        cer = evaluate.load("cer").compute(predictions=pred_str, references=label_str)
        
        return {"cer": cer, "wer": wer}

    def train(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Train the model."""
        # Create training arguments
        training_args = self._create_training_args()
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_processor.prepare_batch,
            compute_metrics=self._compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )
        
        # Save processor
        self.processor.save_pretrained("best_model")
        
        # Start training
        print("\n\n Training STARTED..\n\n")
        train_result = trainer.train()
        
        return train_result 
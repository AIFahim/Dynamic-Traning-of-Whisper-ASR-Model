import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from config import Config
from redis_manager import RedisManager
from data_processor import DataProcessor
from training_manager import TrainingManager
from train_val_df_gen import Train_Val_df

def main():
    # Load configuration
    config = Config()
    
    # Initialize managers
    redis_manager = RedisManager(config.redis)
    data_processor = DataProcessor(None, config.audio_augmentation)  # Processor will be set by training manager
    
    # Load dataset
    print("\n\n Loading Datasets...this might take a while..\n\n")
    with open('/home/asif/merged_dict.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    train_df, val_df, split_index = Train_Val_df.generate_df_from_json(json_data)
    print("Total Datapoint Lengths", len(train_df), len(val_df))
    
    # Create dummy dataset for training
    num_data_points = split_index * 2
    data_list = [('path{}.flac'.format(i), 'text{}'.format(i)) for i in range(1, num_data_points+1)]
    flac_path, txt_list = zip(*data_list)
    data = {
        'input_features': flac_path,
        'labels': txt_list,
    }
    dummy_train_df = pd.DataFrame(data)
    
    # Create dataset dictionary
    dataset = DatasetDict({
        "train": Dataset.from_pandas(dummy_train_df),
        "test": Dataset.from_pandas(val_df)
    })
    
    # Initialize training manager
    training_manager = TrainingManager(
        model_config=config.model,
        data_processor=data_processor,
        redis_manager=redis_manager
    )
    
    # Set processor in data processor
    data_processor.processor = training_manager.processor
    
    # Start training
    train_result = training_manager.train(dataset["train"], dataset["test"])
    
    return train_result

if __name__ == "__main__":
    main() 
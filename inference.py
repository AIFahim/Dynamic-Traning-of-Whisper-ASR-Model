import os
import glob
import warnings
from pathlib import Path
from transformers import pipeline

warnings.filterwarnings("ignore")

class BengaliASRInference:
    def __init__(self, 
                 asr_model_path,
                 chunk_length_s=20.1,
                 enable_beam=True,
                 device=0):
        """
        Initialize the Bengali ASR inference pipeline
        
        Args:
            asr_model_path: Path to the ASR model
            chunk_length_s: Length of audio chunks to process
            enable_beam: Whether to use beam search
            device: GPU device ID
        """
        self.chunk_length_s = chunk_length_s
        self.enable_beam = enable_beam
        self.batch_size = 4 if enable_beam else 8
        self.device = device
        
        # Initialize ASR pipeline with separate model and tokenizer
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=asr_model_path,
            tokenizer="openai/whisper-medium",  # Use the base Whisper medium tokenizer
            chunk_length_s=chunk_length_s,
            device=device,
            batch_size=self.batch_size
        )
        self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(
            language="bn", 
            task="transcribe"
        )
        
    def fix_repetition(self, text, max_count=8):
        """Remove repeated words that exceed max_count"""
        uniq_word_counter = {}
        words = text.split()
        for word in text.split():
            if word not in uniq_word_counter:
                uniq_word_counter[word] = 1
            else:
                uniq_word_counter[word] += 1

        for word, count in uniq_word_counter.items():
            if count > max_count:
                words = [w for w in words if w != word]
        return " ".join(words)
    
    def process_audio(self, audio_path):
        """Process a single audio file"""
        if self.enable_beam:
            result = self.pipe(
                audio_path,
                generate_kwargs={"max_length": 260, "num_beams": 4}
            )
        else:
            result = self.pipe(audio_path)
            
        text = result['text'].strip()
        text = self.fix_repetition(text)
        return text
    
    def process_directory(self, input_dir):
        """Process all audio files in a directory and return transcriptions"""
        files = list(glob.glob(os.path.join(input_dir, '*.wav')))
        files += list(glob.glob(os.path.join(input_dir, '*.mp3')))
        files.sort()
        
        transcriptions = {}
        for f in files:
            file_id = Path(f).stem
            text = self.process_audio(f)
            transcriptions[file_id] = text
                
        return transcriptions

def main():
    # Use the user's model from Hugging Face
    MODEL = "AIFahim/900hr_plus_augmented_whisper_medium"
    
    # Initialize inference
    asr = BengaliASRInference(
        asr_model_path=MODEL,
        chunk_length_s=20.1,
        enable_beam=True
    )
    
    # Process files
    input_dir = './audio_input'
    transcriptions = asr.process_directory(input_dir)
    
    # Print results
    for file_id, text in transcriptions.items():
        print(f"\nFile: {file_id}")
        print(f"Transcription: {text}")
    
    print("\nInference finished!")

if __name__ == "__main__":
    main() 
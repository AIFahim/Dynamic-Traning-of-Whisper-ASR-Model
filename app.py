import streamlit as st
import os
from pathlib import Path
from inference import BengaliASRInference
import tempfile

# Set page config
st.set_page_config(
    page_title="Bengali ASR Transcription",
    page_icon="🎤",
    layout="wide"
)

# Initialize the ASR model
@st.cache_resource
def load_model():
    MODEL = "AIFahim/900hr_plus_augmented_whisper_medium"
    return BengaliASRInference(
        asr_model_path=MODEL,
        chunk_length_s=20.1,
        enable_beam=True
    )

def main():
    st.title("🎤 Bengali Speech Recognition")
    st.markdown("""
    This application transcribes Bengali audio files using the Whisper ASR model.
    Upload your audio file (WAV or MP3) and get the transcription.
    """)

    # Load the model
    try:
        asr = load_model()
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Display audio player
        st.audio(uploaded_file)

        # Process button
        if st.button("Transcribe Audio"):
            with st.spinner("Processing audio..."):
                try:
                    # Process the audio file
                    transcription = asr.process_audio(tmp_file_path)
                    
                    # Display results
                    st.subheader("Transcription Result")
                    st.write(transcription)
                    
                    # Add download button for transcription
                    st.download_button(
                        label="Download Transcription",
                        data=transcription,
                        file_name=f"{Path(uploaded_file.name).stem}_transcription.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error during transcription: {str(e)}")
                finally:
                    # Clean up temporary file
                    os.unlink(tmp_file_path)

    # Add some information about the model
    with st.expander("About the Model"):
        st.markdown("""
        This application uses the following model:
        - **Model Name**: AIFahim/900hr_plus_augmented_whisper_medium
        - **Type**: Whisper ASR
        - **Language**: Bengali
        - **Features**:
            - Beam search enabled
            - Chunk length: 20.1 seconds
            - Automatic repetition removal
        """)

if __name__ == "__main__":
    main() 
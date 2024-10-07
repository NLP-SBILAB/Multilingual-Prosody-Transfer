import argparse
from datasets import load_dataset, Audio
import librosa
import soundfile as sf
from TTS.api import TTS

# Configuration for datasets
datasets = {
    'voxpopuli': ['fr', 'de', 'es', 'nl'],
    'commonvoice': ['hi', 'ta']
}

def load_and_process_dataset(source, language, split='train'):
    if source == 'voxpopuli':
        dataset = load_dataset("facebook/voxpopuli", language, split=split)
    elif source == 'commonvoice':
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", language, split=split)
    
    # Ensure the audio is at the correct sample rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset

def preprocess_audio(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Save processed audio for voice cloning
    processed_audio_path = "processed_audio.wav"
    sf.write(processed_audio_path, y, sr)
    return processed_audio_path

def perform_voice_cloning(text, processed_audio_path, target_language_iso):
    # Initialize the TTS API with the appropriate model
    api = TTS(f"tts_models/{target_language_iso}/fairseq/vits")

    # Output file path
    output_path = f"output_{target_language_iso}.wav"

    # Perform TTS with voice cloning to file
    api.tts_with_vc_to_file(
        text=text,
        speaker_wav=processed_audio_path,
        file_path=output_path
    )
    return output_path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Voice Cloning and TTS System")
    parser.add_argument("--source", choices=['voxpopuli', 'commonvoice'], required=True, help="Choose the source dataset")
    parser.add_argument("--language", type=str, required=True, help="Specify the language code")
    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.language not in datasets[args.source]:
        raise ValueError(f"The language {args.language} is not available for the {args.source} dataset.")

    dataset = load_and_process_dataset(args.source, args.language)

    # Assuming the first entry for demonstration
    audio_path = dataset[0]['audio']['path']
    text = dataset[0].get('sentence', dataset[0].get('normalized_text', ''))
    processed_audio_path = preprocess_audio(audio_path)

    # Generate voice cloned audio with prosody transfer
    output_path = perform_voice_cloning(text, processed_audio_path, args.language)
    print(f"Generated voice cloned audio with prosody at: {output_path}")

if __name__ == "__main__":
    main()
import argparse
import numpy as np
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def load_audio(audio_path, sr=16000):
    """ Load audio file with librosa. """
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio

def compute_mfcc(audio, sr=16000, n_mfcc=13):
    """ Compute the Mel-frequency cepstral coefficients (MFCCs) of the audio. """
    mfcc = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def calculate_mcd(mfcc1, mfcc2):
    """ Calculate the Mel Cepstral Distortion (MCD) using Dynamic Time Warping. """
    distance, path = fastdtw(mfcc1.T, mfcc2.T, dist=euclidean)
    # Calculate mean of distances for all matched frames
    mcd = np.mean([euclidean(mfcc1[:, p[0]], mfcc2[:, p[1]]) for p in path])
    # Conversion factor for MCD
    mcd = mcd * (10 / np.log(10)) * np.sqrt(2)
    return mcd

def parse_arguments():
    """ Setup argparse to handle command line arguments. """
    parser = argparse.ArgumentParser(description='Calculate the Mel Cepstral Distortion (MCD) between two audio clips.')
    parser.add_argument('audio_path1', type=str, help='Path to the first audio file.')
    parser.add_argument('audio_path2', type=str, help='Path to the second audio file.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Load audio files
    audio1 = load_audio(args.audio_path1)
    audio2 = load_audio(args.audio_path2)
    
    # Compute MFCCs
    mfcc1 = compute_mfcc(audio1)
    mfcc2 = compute_mfcc(audio2)
    
    # Calculate MCD
    mcd = calculate_mcd(mfcc1, mfcc2)
    print(f"The Mel Cepstral Distortion between the two audio files is: {mcd:.2f} dB")

if __name__ == '__main__':
    main()
import os

import librosa
import python_speech_features
import parselmouth

from scipy.signal.windows import hamming
import numpy as np

import config

# lembrete importante:
# Rescaling the data to small values (in general, input values to a neural network should be close 
# to zero -- typically we expect either data with zero-mean and unit-variance, or data in the [0, 1] range).
# o keras tem modulo de normalization. talvez seja necessario usar...?

def apply_preemphasis(audio_signal: np.ndarray):
    return librosa.effects.preemphasis(audio_signal, coef=config.PREEMPHASIS_COEFFICIENT)

def extract_features(audio_signal: np.ndarray):
    # extracts jitter, shimmer and MFCCs

    mfcc = python_speech_features.mfcc(
        signal=audio_signal, 
        samplerate=config.DEFAULT_SAMPLE_RATE, 
        winlen=config.FRAME_LENGTH / config.DEFAULT_SAMPLE_RATE, 
        winstep=config.HOP_LENGTH / config.DEFAULT_SAMPLE_RATE,
        numcep=config.NUM_MFCC, 
        nfilt=config.N_MELS, 
        nfft=config.FRAME_LENGTH, 
        lowfreq=config.F_MIN, 
        highfreq=config.F_MAX,
        preemph=config.PREEMPHASIS_COEFFICIENT, 
        ceplifter=config.CEPLIFTER, 
        appendEnergy=True, 
        winfunc=hamming
    )

    snd = parselmouth.Sound(f"{config.DATASET_PATH}/patologicas/carcinoma_masculino_1.wav")

    pitch = snd.to_pitch()
    pulses = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")

    # jitter. (l, r, period floor, period ceiling, maximum period factor)
    jitter_local = parselmouth.praat.call(pulses, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)

    # shimmer
    shimmer_local = parselmouth.praat.call([snd, pulses], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)

    # https://bibliotecadigital.ipb.pt/bitstream/10198/20502/1/pauta-relatorio-43.pdf
    # calculates jitter and shimmer as "absolute" metrics, for the whole audio signal, not its segments

    return (mfcc, jitter_local, shimmer_local)

def pre_processing(audio_signal: np.ndarray) -> np.ndarray:
    # apply pre emphasis and hamming window function
    processed_signal = apply_preemphasis(audio_signal)

    frames = librosa.util.frame(processed_signal, frame_length=config.FRAME_LENGTH, hop_length=config.HOP_LENGTH)
    windowed_frames = np.hamming(config.FRAME_LENGTH).reshape(-1, 1) * frames

    # overlapping frames with window function applied
    return windowed_frames

# validates whether the sample rate across all the dataset is unique
def validate_sample_rate():
    sample_rates = {}

    for (dirpath, _, filenames) in os.walk(config.DATASET_PATH):
        for f in filenames:
            _, sr = librosa.load(f"{dirpath}/{f}", sr=None)
            sample_rates[sr] = True

    if len(sample_rates) > 1:
        raise Exception("Sample rate is not unique in the dataset.")

if __name__ == "__main__":
    validate_sample_rate()

    # librosa -> load. keeps default sample rate with sr=None. each value is a sample (amplitude).

    y, _ = librosa.load(f"{config.DATASET_PATH}/patologicas/carcinoma_masculino_1.wav", sr=None)
    y2, _ = librosa.load(f"{config.DATASET_PATH}/patologicas/carcinoma_masculino_1.wav", sr=None)
    
    extract_features(y)
    pre_processing(y2)

    # still need to know how it works to input the data in the NN

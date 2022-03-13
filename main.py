import os

import librosa
import numpy as np

from python_speech_features.base import mfcc
from scipy.signal.windows import hamming
from tensorflow.keras import preprocessing

from config import (
    CEPLIFTER,
    DATASET_PATH,
    DEFAULT_SAMPLE_RATE,
    F_MAX,
    F_MIN,
    FRAME_LENGTH,
    HOP_LENGTH,
    MASK_VALUE,
    N_MELS,
    NUM_MFCC,
    PREEMPHASIS_COEFFICIENT
)


def apply_preemphasis(audio_signal: np.ndarray):
    # applies librosa's preemphasis with the coefficient defined on config
    return librosa.effects.preemphasis(
        audio_signal,
        coef=PREEMPHASIS_COEFFICIENT)


def extract_features(audio_signal: np.ndarray, full_path: str):
    # extracts MFCCs from audio signal
    mfccs = mfcc(
        signal=audio_signal,
        samplerate=DEFAULT_SAMPLE_RATE,
        winlen=FRAME_LENGTH / DEFAULT_SAMPLE_RATE,
        winstep=HOP_LENGTH / DEFAULT_SAMPLE_RATE,
        numcep=NUM_MFCC,
        nfilt=N_MELS,
        nfft=FRAME_LENGTH,
        lowfreq=F_MIN,
        highfreq=F_MAX,
        preemph=PREEMPHASIS_COEFFICIENT,
        ceplifter=CEPLIFTER,
        appendEnergy=True,
        winfunc=hamming
    )

    return mfccs.flatten()


def pre_processing(audio_signal: np.ndarray) -> np.ndarray:
    # apply pre emphasis and hamming window function
    processed_signal = apply_preemphasis(audio_signal)

    frames = librosa.util.frame(
        processed_signal,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH)

    windowed_frames = np.hamming(FRAME_LENGTH).reshape(-1, 1) * frames

    # overlapping frames with window function applied
    return windowed_frames.flatten()


def validate_sample_rate():
    # validates whether the sample rate across all the dataset is unique
    sample_rates = {}

    for (dirpath, _, filenames) in os.walk(DATASET_PATH):
        for f in filenames:
            _, sr = librosa.load(f"{dirpath}/{f}", sr=None)
            sample_rates[sr] = True

    if len(sample_rates) > 1:
        raise Exception("Sample rate is not unique in the dataset.")


def load_dataset():
    dataset = []
    results = []

    for (dirpath, _, filenames) in os.walk(DATASET_PATH):
        for f in filenames:
            full_path = f"{dirpath}/{f}"
            out, _ = librosa.load(full_path, sr=None)

            sample = pre_processing(out)
            dataset.append(sample)

            if f.find("saudavel") != -1:
                results.append(0)
            else:
                results.append(1)

    padded_samples = preprocessing.sequence.pad_sequences(
        dataset,
        padding="post",
        value=MASK_VALUE  # default value (zero) seems unsuitable in this case
    )

    return (np.asarray(padded_samples), np.asarray(results))


def load_dataset_with_features():
    dataset = []
    results = []

    for (dirpath, _, filenames) in os.walk(DATASET_PATH):
        for f in filenames:
            full_path = f"{dirpath}/{f}"

            # default sr with sr=None. each value is a sample (amplitude)
            out, _ = librosa.load(full_path, sr=None)

            mfccs = extract_features(out, full_path)
            dataset.append(mfccs)

            if f.find("saudavel") != -1:
                results.append(0)
            else:
                results.append(1)

    padded_mfccs = preprocessing.sequence.pad_sequences(
        dataset,
        padding="post",
        value=MASK_VALUE
    )

    return (np.asarray(padded_mfccs), np.asarray(results))

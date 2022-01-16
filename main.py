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
    N_MELS,
    NUM_MFCC,
    PREEMPHASIS_COEFFICIENT
)

"""
lembrete importante:
Rescaling the data to small values (in general, input values to a neural
network should be close to zero -- typically we expect either data with
zero-mean and unit-variance, or data in the [0, 1] range). o keras tem modulo
de normalization. talvez seja necessario usar...?
"""


def apply_preemphasis(audio_signal: np.ndarray):
    return librosa.effects.preemphasis(
        audio_signal,
        coef=PREEMPHASIS_COEFFICIENT)


def extract_features(audio_signal: np.ndarray):
    # extracts jitter, shimmer and MFCCs

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

    # # windows (len is number of windows extracted. depends on audio size)
    # print("bg", mfcc)
    # print(len(mfcc))

    # # mfccs for a window (len is NUM_MFCC)
    # print(mfcc[0])
    # print(len(mfcc[0]))

    # snd = parselmouth.Sound(
    #     f"{DATASET_PATH}/pathological/carcinoma_masculino_1.wav")

    # pitch = snd.to_pitch()
    # pulses = parselmouth.praat.call([snd, pitch], "To PointProcess (cc)")

    # # jitter. (l, r, period floor, period ceiling, maximum period factor)
    # jitter_local = parselmouth.praat.call(
    #     pulses,
    #     "Get jitter (local)",
    #     0.0,
    #     0.0,
    #     0.0001,
    #     0.02,
    #     1.3)

    # shimmer
    # shimmer_local = parselmouth.praat.call(
    #     [snd, pulses],
    #     "Get shimmer (local)",
    #     0.0,
    #     0.0,
    #     0.0001,
    #     0.02,
    #     1.3,
    #     1.6)

    # https://bibliotecadigital.ipb.pt/bitstream/10198/20502/1/pauta-relatorio-43.pdf
    # calculates jitter and shimmer as "absolute" metrics, for the whole audio
    # signal, not its segments

    return mfccs  # jitter_local, shimmer_local off the table for now


def pre_processing(audio_signal: np.ndarray) -> np.ndarray:
    # apply pre emphasis and hamming window function
    processed_signal = apply_preemphasis(audio_signal)

    frames = librosa.util.frame(
        processed_signal,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH)

    windowed_frames = np.hamming(FRAME_LENGTH).reshape(-1, 1) * frames

    # overlapping frames with window function applied
    return windowed_frames


# validates whether the sample rate across all the dataset is unique
def validate_sample_rate():
    sample_rates = {}

    for (dirpath, _, filenames) in os.walk(DATASET_PATH):
        for f in filenames:
            _, sr = librosa.load(f"{dirpath}/{f}", sr=None)
            sample_rates[sr] = True

    if len(sample_rates) > 1:
        raise Exception("Sample rate is not unique in the dataset.")


def normalize_input(dataset):
    maxi = 0

    for i in dataset:
        for j in i:
            maxi = max(maxi, abs(j))

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] /= maxi

    return dataset


def load_dataset():
    dataset = []
    results = []

    for (dirpath, _, filenames) in os.walk(DATASET_PATH):
        for f in filenames:
            out, _ = librosa.load(f"{dirpath}/{f}", sr=None)

            # 1600 x 23 (mini)
            sample = pre_processing(out)

            # assuring every input will have the same number of windows
            sample = np.delete(sample, slice(23, len(sample)), 1)
            sample = sample.flatten()

            dataset.append(sample)

            if f.find("saudavel") != -1:
                results.append(0)
            else:
                results.append(1)

    # normalized_dataset = normalize_input(dataset)
    # padded_samples = preprocessing.sequence.pad_sequences(
    #     dataset,
    #     padding="post",
    # )

    return (np.asarray(dataset), np.asarray(results))


def load_dataset_with_features():
    dataset = []
    results = []

    for (dirpath, _, filenames) in os.walk(DATASET_PATH):
        for f in filenames:
            out, _ = librosa.load(f"{dirpath}/{f}", sr=None)

            # max: 274, min: 24. is there a null value for mfccs? to pad
            mfccs = extract_features(out)

            # assuring every input will have the same number of windows
            # mfccs = np.delete(mfccs, slice(24, len(mfccs)), 0)
            mfccs = mfccs.flatten()

            dataset.append(mfccs)

            if f.find("saudavel") != -1:
                results.append(0)
            else:
                results.append(1)

    # normalized_dataset = normalize_input(dataset)
    padded_mfccs = preprocessing.sequence.pad_sequences(
        dataset,
        padding="post",
    )

    return (np.asarray(padded_mfccs), np.asarray(results))


if __name__ == "__main__":
    validate_sample_rate()

    # librosa -> load. keeps default sample rate with sr=None. each value is
    # a sample (amplitude).

    y, _ = librosa.load(
        f"{DATASET_PATH}/pathological/carcinoma_masculino_1.wav",
        sr=None)

    y2, _ = librosa.load(
        f"{DATASET_PATH}/pathological/carcinoma_masculino_1.wav",
        sr=None)

    extract_features(y)
    pre_processing(y2)

    print(load_dataset_with_features())

    # still need to know how it works to input the data in the NN

DATASET_PATH = "path/to/dataset"
PREEMPHASIS_COEFFICIENT = 0.95  # 0.95 ref: vivlian

NUM_MFCC = 13
N_CLUSTERS = 64
DEFAULT_SAMPLE_RATE = 50000

# Segmentation of 32ms. ref: vivlian
# 1seg = 50000 samples
# 0,032 seg = 50000 x 0,032 = 1600
n_samples_on_window = int(0.032 * DEFAULT_SAMPLE_RATE)

# also used at n_fft (number of samples on window applied on FFT when
# extracting MFCCs)
FRAME_LENGTH = n_samples_on_window

# 50% overlap = 20ms / 2 = 10ms
n_samples_on_overlap = int(n_samples_on_window / 2)

HOP_LENGTH = n_samples_on_overlap

F_MIN = 0
F_MAX = None
N_MELS = 40
CEPLIFTER = 0

# k=10 ref: lucas
K_VALUE = 10

N_EPOCHS = 40
MASK_VALUE = -1

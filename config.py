DATASET_PATH = "/Users/matheus/workspace/uni/tcc/dataset"
PREEMPHASIS_COEFFICIENT = 0.95  # ref: vivlian

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

# overlapping of 50% = 20ms / 2 = 10ms
n_samples_on_overlap = int(n_samples_on_window / 2)

HOP_LENGTH = n_samples_on_overlap

F_MIN = 0
F_MAX = None
N_MELS = 40
CEPLIFTER = 0

K_VALUE = 5
MASK_VALUE = -1

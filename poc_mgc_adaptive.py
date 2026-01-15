"""
POC: Adaptive Denoising Preprocessing for Music Genre Classification (MGC)

This script performs a controlled experiment comparing a baseline
preprocessing pipeline (no denoising) vs an adaptive denoising pipeline
that selects a denoiser based on a simple spectral-flatness heuristic.

Notes:
- Downloads and extracts GTZAN automatically.
- Creates a noisy dataset: 1/3 hiss, 1/3 rumble, 1/3 clean.
- Implements simple spectral-subtraction and median-filter denoisers.
- Builds a small CNN and runs two experiments (baseline vs adaptive).

This is a POC — training can be slow depending on your machine. You
can reduce `MAX_SAMPLES` or `EPOCHS` to iterate faster.
"""
import os
import glob
import random
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import signal
from scipy.signal import medfilt2d

# ----------------------------- CONFIG ---------------------------------
SR = 22050
DURATION = 30  # GTZAN clips are 30s
SAMPLES = SR * DURATION
N_MELS = 128
MEL_FRAMES = 128  # target frames for spectrograms (time axis)
N_FFT = 2048
HOP_LENGTH = 512

# Controls to make the POC runnable quickly during development
MAX_SAMPLES = None  # set to e.g. 60-300 to limit dataset size (or None for all ~1000 clips)
EPOCHS = 6
BATCH_SIZE = 16

# GTZAN origin (common mirror)
GTZAN_URL = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'


# -------------------------- DATA / NOISE HELPERS ----------------------
def add_hiss_noise(audio, snr_db=15):
    """Add stationary white (hiss) noise at approx snr_db (lower = more noise)."""
    rms = np.sqrt(np.mean(audio**2)) + 1e-8
    # compute noise RMS from desired SNR
    noise_rms = rms / (10**(snr_db / 20.0))
    noise = np.random.normal(0, 1, size=len(audio))
    noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-8) * noise_rms
    return audio + noise


def add_rumble_noise(audio, snr_db=15, cutoff_hz=300.0):
    """Add low-frequency non-stationary (rumble) noise (lower = more noise).

    We generate brown-ish noise by integrating white noise and then
    lowpass filtering to emphasize low frequencies.
    """
    rms = np.sqrt(np.mean(audio**2)) + 1e-8
    noise = np.cumsum(np.random.randn(len(audio)))
    noise = noise - np.mean(noise)
    # lowpass filter to keep only low frequencies
    b, a = signal.butter(4, cutoff_hz / (SR / 2.0), btype='low')
    noise = signal.filtfilt(b, a, noise)
    noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-8)
    noise_rms = rms / (10**(snr_db / 20.0))
    noise = noise * noise_rms
    return audio + noise


# -------------------------- DENOISERS ---------------------------------
def denoise_spectral_subtraction(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Simple spectral subtraction for stationary hiss-like noise.

    Estimate a noise magnitude spectrum using the median across time
    and subtract it from the signal's STFT magnitude with flooring.
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    # estimate noise profile as the median across time bins
    noise_mag = np.median(mag, axis=1, keepdims=True)
    mag_clean = np.maximum(mag - noise_mag, 1e-8)
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length)
    return audio_clean


def denoise_median_filtering(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Median-filtering on magnitude spectrogram to suppress non-stationary rumble."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    # Apply 2D median filtering on the magnitude spectrogram
    mag_med = medfilt2d(mag, kernel_size=(3, 9))
    # Use the median-filtered magnitude for reconstruction
    stft_clean = mag_med * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length)
    return audio_clean


# -------------------------- ADAPTIVE LOGIC ----------------------------
def characterize_noise_type(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, thresh=0.5):
    """Compute a spectral flatness metric and return 'hiss' or 'rumble'.

    Spectral flatness is high for white-noise-like signals. We take the
    mean flatness across frames and compare to a threshold.
    """
    sf = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    mean_sf = float(np.mean(sf))
    if mean_sf > thresh:
        return 'hiss'
    return 'rumble'


# -------------------------- PREPROCESSING PIPELINES -------------------
def audio_to_mel(audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, target_frames=MEL_FRAMES):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to 0..1
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    # Ensure fixed number of frames (time axis) by trimming or padding
    if mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :target_frames]
    return mel_db.astype(np.float32)


def baseline_pipeline(audio_batch):
    X = []
    for audio in audio_batch:
        mel = audio_to_mel(audio)
        X.append(mel)
    X = np.array(X)[..., np.newaxis]
    return X


def adaptive_pipeline(audio_batch):
    X = []
    noise_counts = {'hiss': 0, 'rumble': 0}
    for audio in audio_batch:
        t = characterize_noise_type(audio)
        if t == 'hiss':
            noise_counts['hiss'] += 1
            audio_cleansed = denoise_spectral_subtraction(audio)
        else:
            noise_counts['rumble'] += 1
            audio_cleansed = denoise_median_filtering(audio)
        mel = audio_to_mel(audio_cleansed)
        X.append(mel)
    print(f'  Detected: {noise_counts["hiss"]} hiss, {noise_counts["rumble"]} rumble')
    X = np.array(X)[..., np.newaxis]
    return X


# -------------------------- MODEL ------------------------------------
def build_mgc_cnn(input_shape, num_classes):
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# -------------------------- DATASET CREATION -------------------------
def download_and_prepare_gtzan(dest_dir=None):
    """Check for local dataset first, download only if not found."""
    # Check common local paths first
    local_paths = [
        os.path.join('.', 'datasets', 'genres'),           # ./datasets/genres/
        os.path.join('.', 'datasets', 'Data', 'genres_original'),  # Kaggle structure
        os.path.join('.', 'datasets', 'genres_original'),  # Alternative Kaggle
        os.path.join('.', 'Data', 'genres_original'),      # Another common structure
    ]
    
    for local_path in local_paths:
        if os.path.exists(local_path) and os.path.isdir(local_path):
            # Verify it has genre subdirectories
            subdirs = [d for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d))]
            if len(subdirs) >= 5:  # Should have ~10 genre folders
                print(f'Using local GTZAN dataset from: {local_path}')
                return local_path
    
    # If no local dataset found, download
    print('Local dataset not found, downloading GTZAN...')
    path = tf.keras.utils.get_file('genres', origin=GTZAN_URL, untar=True, cache_dir='.')
    # tf returns the root cache dir; the archive extracts into 'genres'
    extracted = os.path.join(os.path.dirname(path), 'genres')
    if os.path.exists(extracted):
        return extracted
    # Fallback: if the above path is the extracted folder
    if os.path.isdir(path):
        return path
    raise RuntimeError('Could not find/extract GTZAN archive')


def create_noisy_dataset(genres_root, max_samples=None):
    files = glob.glob(os.path.join(genres_root, '*', '*.wav'))
    files.sort()
    if max_samples is not None:
        files = files[:max_samples]
    random.seed(0)
    random.shuffle(files)

    X = []
    y = []
    skipped = 0
    for i, f in enumerate(files):
        try:
            audio, sr = librosa.load(f, sr=SR, duration=DURATION)
            # Decide noise application: first third hiss, second third rumble, last third clean
            idx = i % 3
            if idx == 0:
                audio_noisy = add_hiss_noise(audio)
            elif idx == 1:
                audio_noisy = add_rumble_noise(audio)
            else:
                audio_noisy = audio
            genre = os.path.basename(os.path.dirname(f))
            X.append(audio_noisy)
            y.append(genre)
            
            if (i + 1) % 100 == 0:
                print(f'Processed {i + 1}/{len(files)} files...')
        except Exception as e:
            skipped += 1
            print(f'Skipping corrupted file {os.path.basename(f)}: {str(e)[:50]}')
            continue

    print(f'Loaded {len(X)} files, skipped {skipped} corrupted files')
    
    # Map genres to ints
    genres = sorted(list(set(y)))
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    y_idx = np.array([genre_to_idx[g] for g in y], dtype=np.int32)
    return X, y_idx, genres

 
# -------------------------- EXPERIMENT --------------------------------
def run_experiment():
    print('Downloading GTZAN (this may take a while)...')
    genres_root = download_and_prepare_gtzan()
    print('Preparing noisy dataset...')
    X_audio, y, genres = create_noisy_dataset(genres_root, max_samples=MAX_SAMPLES)

    # Split into train/test keeping stratification
    X_train_audio, X_test_audio, y_train, y_test = train_test_split(X_audio, y, test_size=0.2, stratify=y, random_state=0)

    print('Creating baseline features...')
    X_train_baseline = baseline_pipeline(X_train_audio)
    X_test_baseline = baseline_pipeline(X_test_audio)

    print('Creating adaptive features...')
    X_train_adaptive = adaptive_pipeline(X_train_audio)
    X_test_adaptive = adaptive_pipeline(X_test_audio)

    input_shape = X_train_baseline.shape[1:]
    num_classes = len(genres)

    # Experiment 1: Baseline
    print('\n=== Experiment 1: Baseline Pipeline ===')
    model_base = build_mgc_cnn(input_shape, num_classes)
    model_base.fit(X_train_baseline, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    preds_base = np.argmax(model_base.predict(X_test_baseline), axis=1)
    acc_base = accuracy_score(y_test, preds_base)

    # Experiment 2: Adaptive
    print('\n=== Experiment 2: Adaptive Pipeline ===')
    model_adapt = build_mgc_cnn(input_shape, num_classes)
    model_adapt.fit(X_train_adaptive, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    preds_adapt = np.argmax(model_adapt.predict(X_test_adaptive), axis=1)
    acc_adapt = accuracy_score(y_test, preds_adapt)

    print('\n=== Summary ===')
    print(f'Baseline Model Test Accuracy: {acc_base:.4f}')
    print(f'Adaptive Framework Model Test Accuracy: {acc_adapt:.4f}')
    if acc_adapt > acc_base:
        print('Conclusion: Adaptive framework improved performance on this run.')
    elif acc_adapt < acc_base:
        print('Conclusion: Adaptive framework did not improve performance on this run.')
    else:
        print('Conclusion: Both pipelines performed equally on this run.')


if __name__ == '__main__':
    run_experiment()

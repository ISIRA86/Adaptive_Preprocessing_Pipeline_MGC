"""
POC: Adaptive Denoising for AudioSet Dataset

AudioSet contains 2M+ 10-second clips from YouTube with real-world noise.
This POC uses a subset focusing on music-related classes.

Dataset: https://research.google.com/audioset/
For this POC, we'll use the balanced train set (22k clips, ~10GB)

Note: AudioSet requires youtube-dl and proper setup. 
Alternative: Use pre-downloaded AudioSet from Kaggle or academic sources.
"""
import os
import glob
import random
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy import signal
from scipy.signal import medfilt2d
import csv

# ----------------------------- CONFIG ---------------------------------
SR = 16000  # AudioSet standard
DURATION = 10  # AudioSet clips are 10 seconds
N_MELS = 128
MEL_FRAMES = 256  # More frames for 10s at 16kHz
N_FFT = 2048
HOP_LENGTH = 160  # ~100 fps for 16kHz

MAX_SAMPLES = None  # None for all available
EPOCHS = 10
BATCH_SIZE = 32

# AudioSet paths
AUDIOSET_DIR = os.path.join('.', 'datasets', 'audioset')
AUDIOSET_CSV = os.path.join('.', 'datasets', 'audioset', 'balanced_train_segments.csv')

# Focus on music/speech classes (mid-level ontology IDs)
TARGET_CLASSES = {
    '/m/04rlf': 'Music',
    '/m/09x0r': 'Speech',
    '/m/0l14md': 'Singing',
    '/m/05148p4': 'Musical instrument',
    '/m/0y4f8': 'Vocal music',
}


# -------------------------- DENOISERS ---------------------------------
def denoise_spectral_subtraction(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Spectral subtraction for stationary background noise."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    
    # Estimate noise from quietest 15% of frames
    frame_energy = np.sum(mag, axis=0)
    noise_threshold = np.percentile(frame_energy, 15)
    noise_frames = frame_energy < noise_threshold
    
    if np.sum(noise_frames) > 5:
        noise_mag = np.mean(mag[:, noise_frames], axis=1, keepdims=True)
    else:
        noise_mag = np.percentile(mag, 10, axis=1, keepdims=True)
    
    # Over-subtraction with flooring
    alpha = 2.0  # Over-subtraction factor
    mag_clean = np.maximum(mag - alpha * noise_mag, 0.1 * mag)
    
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    return audio_clean


def denoise_temporal_median(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Temporal median filtering for non-stationary transients."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    
    # Apply 2D median filter emphasizing temporal smoothing
    mag_filtered = medfilt2d(mag, kernel_size=(3, 15))
    
    stft_clean = mag_filtered * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    return audio_clean


def denoise_harmonic_separation(audio):
    """Separate harmonic (music/speech) from percussive (noise) components."""
    # Harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(audio, margin=3.0)
    
    # Emphasize harmonic content, suppress percussive noise
    # Mix: 90% harmonic, 10% percussive
    audio_clean = 0.9 * y_harmonic + 0.1 * y_percussive
    return audio_clean


# -------------------------- ADAPTIVE LOGIC ----------------------------
def characterize_audio_content(audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Analyze audio to determine optimal denoising strategy."""
    # Spectral features
    sf = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    sc = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    sb = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # Temporal features
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)
    
    # Statistics
    mean_sf = float(np.mean(sf))
    std_sf = float(np.std(sf))
    mean_sc = float(np.mean(sc))
    std_rms = float(np.std(rms))
    mean_zcr = float(np.mean(zcr))
    
    # Decision tree for denoising strategy
    if mean_sf > 0.7:  # Very flat = broadband noise
        return 'spectral_sub'
    elif std_rms > 0.15:  # High amplitude variation = transients
        return 'temporal_median'
    elif mean_sc > 3000 and std_sf < 0.2:  # Harmonic content present
        return 'harmonic_sep'
    else:  # General case
        return 'spectral_sub'


# -------------------------- PREPROCESSING PIPELINES -------------------
def audio_to_mel(audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, target_frames=MEL_FRAMES):
    """Convert audio to mel-spectrogram."""
    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))
    
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    # Resize to target frames
    if mel_db.shape[1] < target_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - mel_db.shape[1])), mode='edge')
    else:
        mel_db = mel_db[:, :target_frames]
    
    return mel_db.astype(np.float32)


def baseline_pipeline(audio_batch):
    """No denoising."""
    X = [audio_to_mel(audio) for audio in audio_batch]
    return np.array(X)[..., np.newaxis]


def adaptive_pipeline(audio_batch):
    """Adaptive denoising pipeline."""
    X = []
    strategy_counts = {'spectral_sub': 0, 'temporal_median': 0, 'harmonic_sep': 0}
    
    for audio in audio_batch:
        strategy = characterize_audio_content(audio)
        strategy_counts[strategy] += 1
        
        if strategy == 'spectral_sub':
            audio_clean = denoise_spectral_subtraction(audio)
        elif strategy == 'temporal_median':
            audio_clean = denoise_temporal_median(audio)
        else:  # harmonic_sep
            audio_clean = denoise_harmonic_separation(audio)
        
        mel = audio_to_mel(audio_clean)
        X.append(mel)
    
    print(f'  Strategies: Spectral={strategy_counts["spectral_sub"]}, '
          f'Temporal={strategy_counts["temporal_median"]}, '
          f'Harmonic={strategy_counts["harmonic_sep"]}')
    
    return np.array(X)[..., np.newaxis]


# -------------------------- MODEL ------------------------------------
def build_audioset_cnn(input_shape, num_classes):
    """CNN for AudioSet classification."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# -------------------------- DATASET LOADING ---------------------------
def load_audioset_dataset(max_samples=None):
    """
    Load AudioSet audio files.
    
    Expected structure:
    datasets/audioset/
        audio/
            YTID1.wav
            YTID2.wav
            ...
        balanced_train_segments.csv
    """
    if not os.path.exists(AUDIOSET_DIR):
        raise FileNotFoundError(
            f"AudioSet not found at {AUDIOSET_DIR}.\n"
            "Download AudioSet or use pre-processed version from:\n"
            "- https://research.google.com/audioset/download.html\n"
            "- Kaggle: https://www.kaggle.com/datasets/google/audioset\n"
            "Structure: ./datasets/audioset/audio/*.wav"
        )
    
    print('Loading AudioSet files...')
    
    audio_dir = os.path.join(AUDIOSET_DIR, 'audio')
    if not os.path.exists(audio_dir):
        audio_dir = AUDIOSET_DIR  # Flat structure
    
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    
    if len(audio_files) == 0:
        raise FileNotFoundError(
            f"No .wav files found in {audio_dir}.\n"
            "Please download and extract AudioSet audio files."
        )
    
    if max_samples:
        audio_files = audio_files[:max_samples]
    
    print(f'Found {len(audio_files)} audio files')
    
    X = []
    y = []
    skipped = 0
    
    # Simple approach: assign pseudo-labels based on filename or use actual labels if available
    # For this POC, we'll use a simple binary classification: clean vs noisy
    # In real scenario, you'd parse the CSV and map to actual AudioSet labels
    
    for i, filepath in enumerate(audio_files):
        try:
            audio, sr = librosa.load(filepath, sr=SR, duration=DURATION)
            
            # Simplified labeling (replace with actual AudioSet labels)
            # Here we use a simple heuristic: high ZCR = noisy/speech (1), low = music (0)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
            label = 1 if zcr > 0.1 else 0
            
            X.append(audio)
            y.append(label)
            
            if (i + 1) % 500 == 0:
                print(f'  Loaded {i + 1}/{len(audio_files)} files...')
        
        except Exception as e:
            skipped += 1
            if skipped % 50 == 0:
                print(f'  Skipped {skipped} corrupted files')
            continue
    
    print(f'Loaded {len(X)} clips, skipped {skipped}')
    
    return X, np.array(y, dtype=np.int32), ['Music', 'Speech']


# -------------------------- EXPERIMENT --------------------------------
def run_experiment():
    """Run comparative experiment."""
    print('='*70)
    print('AudioSet Adaptive Denoising POC')
    print('='*70)
    
    X_audio, y, classes = load_audioset_dataset(max_samples=MAX_SAMPLES)
    print(f'\nClasses: {classes}')
    print(f'Total samples: {len(X_audio)}')
    
    X_train_audio, X_test_audio, y_train, y_test = train_test_split(
        X_audio, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f'Train: {len(X_train_audio)}, Test: {len(X_test_audio)}')
    
    # Features
    print('\n' + '='*70)
    print('Baseline features...')
    X_train_baseline = baseline_pipeline(X_train_audio)
    X_test_baseline = baseline_pipeline(X_test_audio)
    
    print('\n' + '='*70)
    print('Adaptive features...')
    X_train_adaptive = adaptive_pipeline(X_train_audio)
    X_test_adaptive = adaptive_pipeline(X_test_audio)
    
    input_shape = X_train_baseline.shape[1:]
    num_classes = len(classes)
    
    # Baseline
    print('\n' + '='*70)
    print('BASELINE')
    print('='*70)
    model_base = build_audioset_cnn(input_shape, num_classes)
    model_base.fit(X_train_baseline, y_train, validation_split=0.15, 
                   epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    
    preds_base = np.argmax(model_base.predict(X_test_baseline, verbose=0), axis=1)
    acc_base = accuracy_score(y_test, preds_base)
    
    # Adaptive
    print('\n' + '='*70)
    print('ADAPTIVE')
    print('='*70)
    model_adapt = build_audioset_cnn(input_shape, num_classes)
    model_adapt.fit(X_train_adaptive, y_train, validation_split=0.15,
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    
    preds_adapt = np.argmax(model_adapt.predict(X_test_adaptive, verbose=0), axis=1)
    acc_adapt = accuracy_score(y_test, preds_adapt)
    
    # Results
    print('\n' + '='*70)
    print('RESULTS')
    print('='*70)
    print(f'Baseline:    {acc_base:.4f} ({acc_base*100:.2f}%)')
    print(f'Adaptive:    {acc_adapt:.4f} ({acc_adapt*100:.2f}%)')
    print(f'Improvement: {(acc_adapt-acc_base)*100:+.2f}%')
    
    if acc_adapt > acc_base:
        print('\n✓ Adaptive framework IMPROVED performance!')
    else:
        print('\n✗ No improvement observed.')


if __name__ == '__main__':
    run_experiment()

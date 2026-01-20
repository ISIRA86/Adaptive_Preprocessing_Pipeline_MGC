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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import signal
from scipy.signal import medfilt2d

# ----------------------------- CONFIG ---------------------------------
SR = 22050
DURATION = 29  # FMA clips are 30s, use 29 to be safe
N_MELS = 128
MEL_FRAMES = 128
N_FFT = 2048
HOP_LENGTH = 512

# Subset for faster experimentation
MAX_SAMPLES = None  # None for all, or e.g. 2000 for testing
EPOCHS = 10
BATCH_SIZE = 16

# FMA dataset path
FMA_AUDIO_DIR = os.path.join('.', 'datasets', 'fma_medium')
FMA_METADATA = os.path.join('.', 'datasets', 'fma_metadata', 'tracks.csv')

# Target genres (FMA-small has these 8 genres)
TARGET_GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 
                 'Instrumental', 'International', 'Pop', 'Rock']


# -------------------------- DENOISERS ---------------------------------
def denoise_spectral_subtraction(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Spectral subtraction for stationary noise - GENTLE version."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    
    # Use minimum instead of median for gentler noise estimation
    noise_mag = np.percentile(mag, 5, axis=1, keepdims=True)  # Bottom 5% instead of median
    
    # Gentle subtraction with higher floor
    alpha = 0.5  # Reduced from implicit 1.0 - less aggressive
    mag_clean = np.maximum(mag - alpha * noise_mag, 0.5 * mag)  # Floor at 50% of original
    
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    return audio_clean


def denoise_median_filtering(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Median filtering for non-stationary noise - GENTLE version."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    
    # Smaller kernel - less aggressive smoothing
    mag_med = medfilt2d(mag, kernel_size=(3, 5))  # Reduced from (3, 9)
    
    # Blend with original to preserve more detail
    alpha = 0.6  # 60% filtered, 40% original
    mag_clean = alpha * mag_med + (1 - alpha) * mag
    
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    return audio_clean


def denoise_wiener_filter(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Wiener filtering - adaptive approach for general noise - GENTLE version."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    
    # Estimate noise from quiet portions (bottom 5% of energy, not 10%)
    energy = np.sum(mag, axis=0)
    noise_frames = energy < np.percentile(energy, 5)  # More conservative
    if np.sum(noise_frames) > 0:
        noise_mag = np.median(mag[:, noise_frames], axis=1, keepdims=True)
    else:
        noise_mag = np.percentile(mag, 5, axis=1, keepdims=True)
    
    # Wiener filter with floor: H = (S^2) / (S^2 + N^2), but clip to preserve signal
    signal_power = mag ** 2
    noise_power = noise_mag ** 2
    wiener_gain = signal_power / (signal_power + noise_power + 1e-8)
    wiener_gain = np.clip(wiener_gain, 0.5, 1.0)  # Floor at 50% instead of allowing near-zero
    mag_clean = mag * wiener_gain
    
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    return audio_clean



# -------------------------- ADAPTIVE LOGIC ----------------------------
def characterize_noise_type(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Enhanced noise characterization using multiple features - MORE CONSERVATIVE."""
    # Spectral flatness
    sf = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    mean_sf = float(np.mean(sf))
    
    # Spectral centroid (center of mass of spectrum)
    sc = librosa.feature.spectral_centroid(y=audio, sr=SR, n_fft=n_fft, hop_length=hop_length)
    mean_sc = float(np.mean(sc))
    
    # Zero crossing rate (temporal characteristic)
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)
    mean_zcr = float(np.mean(zcr))
    
    # More conservative thresholds to avoid over-denoising
    if mean_sf > 0.75:  # Much flatter spectrum required (was 0.6)
        return 'broadband'  # Use spectral subtraction
    elif mean_sc < 1500:  # Lower frequency threshold (was 2000)
        return 'lowfreq'  # Use median filtering
    elif mean_zcr > 0.20:  # Higher variability required (was 0.15)
        return 'transient'  # Use median filtering
    else:
        return 'general'  # Use Wiener filtering


# -------------------------- PREPROCESSING PIPELINES -------------------
def audio_to_mel(audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, target_frames=MEL_FRAMES):
    """Convert audio to normalized mel-spectrogram."""
    if len(audio) < sr:  # Too short
        audio = np.pad(audio, (0, sr - len(audio)))
    
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    if mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :target_frames]
    
    return mel_db.astype(np.float32)


def baseline_pipeline(audio_batch):
    """No denoising - direct conversion to mel-spectrograms."""
    X = []
    for audio in audio_batch:
        mel = audio_to_mel(audio)
        X.append(mel)
    return np.array(X)[..., np.newaxis]


def adaptive_pipeline(audio_batch):
    """Adaptive denoising based on noise characteristics."""
    X = []
    noise_stats = {'broadband': 0, 'lowfreq': 0, 'transient': 0, 'general': 0}
    
    for audio in audio_batch:
        noise_type = characterize_noise_type(audio)
        noise_stats[noise_type] += 1
        
        if noise_type == 'broadband':
            audio_clean = denoise_spectral_subtraction(audio)
        elif noise_type in ['lowfreq', 'transient']:
            audio_clean = denoise_median_filtering(audio)
        else:  # general
            audio_clean = denoise_wiener_filter(audio)
        
        mel = audio_to_mel(audio_clean)
        X.append(mel)
    
    print(f'  Noise types: Broadband={noise_stats["broadband"]}, '
          f'LowFreq={noise_stats["lowfreq"]}, Transient={noise_stats["transient"]}, '
          f'General={noise_stats["general"]}')
    
    return np.array(X)[..., np.newaxis], noise_stats


# -------------------------- MODEL ------------------------------------
def build_genre_cnn(input_shape, num_classes):
    """CNN architecture for genre classification."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# -------------------------- DATASET LOADING ---------------------------
def load_fma_metadata():
    """Load FMA metadata CSV."""
    if not os.path.exists(FMA_METADATA):
        raise FileNotFoundError(
            f"FMA metadata not found at {FMA_METADATA}.\n"
            "Download from: https://github.com/mdeff/fma\n"
            "Extract tracks.csv to ./datasets/fma_metadata/"
        )
    
    # FMA CSV has multi-level headers
    tracks = pd.read_csv(FMA_METADATA, index_col=0, header=[0, 1])
    return tracks


def load_fma_dataset(max_samples=None):
    """Load FMA audio files with genre labels."""
    if not os.path.exists(FMA_AUDIO_DIR):
        raise FileNotFoundError(
            f"FMA audio not found at {FMA_AUDIO_DIR}.\n"
            "Download fma_small.zip from: https://github.com/mdeff/fma\n"
            "Extract to ./datasets/fma_small/"
        )
    
    print('Loading FMA metadata...')
    tracks = load_fma_metadata()
    
    # Get genre labels (top-level genre)
    genre_col = ('track', 'genre_top')
    
    X = []
    y = []
    skipped = 0
    
    # Get all audio files
    audio_files = glob.glob(os.path.join(FMA_AUDIO_DIR, '*', '*.mp3'))
    if max_samples:
        audio_files = audio_files[:max_samples]
    
    print(f'Loading {len(audio_files)} audio files...')
    
    for i, filepath in enumerate(audio_files):
        try:
            # Extract track ID from filename
            track_id = int(os.path.splitext(os.path.basename(filepath))[0])
            
            # Get genre from metadata
            if track_id not in tracks.index:
                skipped += 1
                continue
            
            genre = tracks.loc[track_id, genre_col]
            if pd.isna(genre) or genre not in TARGET_GENRES:
                skipped += 1
                continue
            
            # Load audio
            audio, sr = librosa.load(filepath, sr=SR, duration=DURATION, mono=True)
            
            X.append(audio)
            y.append(genre)
            
            if (i + 1) % 500 == 0:
                print(f'  Processed {i + 1}/{len(audio_files)} files, loaded {len(X)} valid tracks...')
        
        except KeyboardInterrupt:
            raise
        except Exception as e:
            skipped += 1
            if skipped % 100 == 0:
                print(f'  Skipped {skipped} files (errors/missing metadata)')
            continue
    
    print(f'Loaded {len(X)} tracks, skipped {skipped} files')
    
    # Map genres to integers
    genre_to_idx = {g: i for i, g in enumerate(sorted(set(y)))}
    y_idx = np.array([genre_to_idx[g] for g in y], dtype=np.int32)
    
    return X, y_idx, list(genre_to_idx.keys())


# -------------------------- EXPERIMENT --------------------------------
def run_experiment():
    """Run comparative experiment: baseline vs adaptive preprocessing."""
    print('='*70)
    print('FMA Adaptive Denoising POC')
    print('='*70)
    
    # Load dataset
    X_audio, y, genres = load_fma_dataset(max_samples=MAX_SAMPLES)
    print(f'\nGenres: {genres}')
    print(f'Total samples: {len(X_audio)}')
    
    # Train/test split
    X_train_audio, X_test_audio, y_train, y_test = train_test_split(
        X_audio, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f'Train: {len(X_train_audio)}, Test: {len(X_test_audio)}')
    
    # Generate features
    print('\n' + '='*70)
    print('Generating baseline features (no denoising)...')
    X_train_baseline = baseline_pipeline(X_train_audio)
    X_test_baseline = baseline_pipeline(X_test_audio)
    
    print('\n' + '='*70)
    print('Generating adaptive features (with adaptive denoising)...')
    X_train_adaptive, train_noise_stats = adaptive_pipeline(X_train_audio)
    X_test_adaptive, test_noise_stats = adaptive_pipeline(X_test_audio)
    
    # Combine noise statistics
    total_noise_stats = {k: train_noise_stats[k] + test_noise_stats[k] 
                         for k in train_noise_stats.keys()}
    
    input_shape = X_train_baseline.shape[1:]
    num_classes = len(genres)
    
    # Experiment 1: Baseline
    print('\n' + '='*70)
    print('EXPERIMENT 1: Baseline Pipeline (No Denoising)')
    print('='*70)
    model_baseline = build_genre_cnn(input_shape, num_classes)
    history_base = model_baseline.fit(
        X_train_baseline, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    
    preds_baseline = np.argmax(model_baseline.predict(X_test_baseline, verbose=0), axis=1)
    acc_baseline = accuracy_score(y_test, preds_baseline)
    
    # Experiment 2: Adaptive
    print('\n' + '='*70)
    print('EXPERIMENT 2: Adaptive Pipeline (With Denoising)')
    print('='*70)
    model_adaptive = build_genre_cnn(input_shape, num_classes)
    history_adapt = model_adaptive.fit(
        X_train_adaptive, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    
    preds_adaptive = np.argmax(model_adaptive.predict(X_test_adaptive, verbose=0), axis=1)
    acc_adaptive = accuracy_score(y_test, preds_adaptive)
    
    # Generate confusion matrices
    cm_baseline = confusion_matrix(y_test, preds_baseline)
    cm_adaptive = confusion_matrix(y_test, preds_adaptive)
    
    # Results
    print('\n' + '='*70)
    print('RESULTS')
    print('='*70)
    print(f'Baseline Test Accuracy:  {acc_baseline:.4f} ({acc_baseline*100:.2f}%)')
    print(f'Adaptive Test Accuracy:  {acc_adaptive:.4f} ({acc_adaptive*100:.2f}%)')
    print(f'Improvement:             {(acc_adaptive - acc_baseline):.4f} ({(acc_adaptive - acc_baseline)*100:.2f}%)')
    
    if acc_adaptive > acc_baseline:
        print('\n✓ Adaptive framework IMPROVED performance!')
    elif acc_adaptive < acc_baseline:
        print('\n✗ Adaptive framework did not improve performance.')
    else:
        print('\n- Both pipelines performed equally.')
    
    print('\nBaseline Classification Report:')
    baseline_report = classification_report(y_test, preds_baseline, target_names=genres, 
                                           zero_division=0, output_dict=True)
    print(classification_report(y_test, preds_baseline, target_names=genres, zero_division=0))
    
    print('\nAdaptive Classification Report:')
    adaptive_report = classification_report(y_test, preds_adaptive, target_names=genres, 
                                           zero_division=0, output_dict=True)
    print(classification_report(y_test, preds_adaptive, target_names=genres, zero_division=0))
    
    # Generate analysis reports
    print('\n' + '='*70)
    print('Generating visualization and analysis reports...')
    print('='*70)
    try:
        from analyze_results import generate_all_reports
        generate_all_reports(
            baseline_acc=acc_baseline,
            adaptive_acc=acc_adaptive,
            baseline_report_dict=baseline_report,
            adaptive_report_dict=adaptive_report,
            genres=genres,
            noise_stats=total_noise_stats,
            baseline_cm=cm_baseline,
            adaptive_cm=cm_adaptive
        )
        print('✓ Reports generated in ./results/ directory')
    except Exception as e:
        print(f'Warning: Could not generate reports: {e}')
        print('Install matplotlib and seaborn: pip install matplotlib seaborn')


if __name__ == '__main__':
    run_experiment()

import os
import glob
import numpy as np
import librosa
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from scipy import signal
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt2d
import warnings
warnings.filterwarnings('ignore')

# Try to import pywt for wavelet denoising (optional)
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("WARNING: pywt not available. Wavelet denoising will be disabled.")


# ----------------------------- CONFIG ---------------------------------
SR = 22050
DURATION = 29
N_MELS = 128
MEL_FRAMES = 128
N_FFT = 2048
HOP_LENGTH = 512

MAX_SAMPLES = 3000 
EPOCHS = 40 
BATCH_SIZE = 32

USE_MULTI_FEATURES = True 

# DATASET SELECTION
DATASET = 'fma_medium'

# Use absolute paths based on this file's location
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FMA_AUDIO_DIR = os.path.join(_PROJECT_ROOT, 'datasets', 'fma_medium')
FMA_METADATA = os.path.join(_PROJECT_ROOT, 'datasets', 'fma_metadata', 'tracks.csv')

TARGET_GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop',
                 'Instrumental', 'International', 'Pop', 'Rock']

# AudioSet music subset (6 genres with clean AudioSet mappings)
AUDIOSET_MUSIC_DIR = os.path.join(_PROJECT_ROOT, 'datasets', 'audioset_music')
AUDIOSET_TARGET_GENRES = ['Electronic', 'Folk', 'Hip-Hop', 'International', 'Pop', 'Rock']

# LUFS target

TARGET_LUFS = -23.0  

# Output directories
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
SPECTROGRAMS_DIR = os.path.join(RESULTS_DIR, 'spectrograms')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SPECTROGRAMS_DIR, exist_ok=True)


# ======================== ADVANCED PREPROCESSING METHODS ==============
# ======================== SPECTRAL GATING ==============================
def denoise_spectral_gating(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                            threshold_db=-40, alpha=0.1):
    try:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(stft), np.angle(stft)
        
        if np.max(mag) < 1e-8:
            return audio
        
        mag_db = librosa.amplitude_to_db(mag, ref=np.max)
        noise_floor = np.percentile(mag_db, 10, axis=1, keepdims=True)
        
        signal_above_noise = mag_db - noise_floor
        gate = 1.0 / (1.0 + np.exp(-alpha * (signal_above_noise - threshold_db)))
        
        mag_gated = mag * gate
        stft_clean = mag_gated * np.exp(1j * phase)
        audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
        
        if not np.isfinite(audio_clean).all():
            return audio
        
        return audio_clean
    except Exception as e:
        print(f"Warning: Spectral gating failed ({e}), returning original audio")
        return audio




# ======================== LUFS NORMALIZATION =====================
def normalize_lufs(audio, target_lufs=TARGET_LUFS, sr=SR):
    try:
        import pyloudnorm as pyln
        
        # Check for invalid audio first
        if not np.isfinite(audio).all():
            return audio
        
        # Check if audio is too quiet (avoid division by zero)
        if np.abs(audio).max() < 1e-8:
            return audio
        
        meter = pyln.Meter(sr)
        current_loudness = meter.integrated_loudness(audio)
        
        # Check if loudness measurement is valid
        if not np.isfinite(current_loudness) or current_loudness < -70:
            # Audio is too quiet, just return normalized by peak
            peak = np.abs(audio).max()
            if peak > 0:
                return audio / peak * 0.1
            return audio
        
        # Normalize to target
        normalized_audio = pyln.normalize.loudness(audio, current_loudness, target_lufs)
        
        # Safety check: ensure output is finite
        if not np.isfinite(normalized_audio).all():
            return audio
        
        # Clip to prevent distortion
        normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
        
        return normalized_audio
    
    except Exception as e:
        # If anything goes wrong, return original audio
        print(f"Warning: LUFS normalization failed ({e}), returning original audio")
        return audio




# ======================== ADAPTIVE ROUTING LOGIC ======================
def characterize_audio_type(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    try:
        analysis_audio = audio[:SR * 10] if len(audio) > SR * 10 else audio
        
        # Spectral flatness (tonal vs noise-like)
        sf = librosa.feature.spectral_flatness(y=analysis_audio, n_fft=n_fft, hop_length=hop_length)
        mean_sf = float(np.mean(sf))
        
        # Harmonic-to-noise ratio (use margin parameter to reduce memory)
        harmonic, percussive = librosa.effects.hpss(analysis_audio, margin=3.0)
        harmonic_energy = np.sum(harmonic ** 2)
        percussive_energy = np.sum(percussive ** 2)
        total_energy = harmonic_energy + percussive_energy
        
        harmonic_ratio = harmonic_energy / (total_energy + 1e-8)
        
        # Spectral centroid (frequency content)
        sc = librosa.feature.spectral_centroid(y=analysis_audio, sr=SR, n_fft=n_fft, hop_length=hop_length)
        mean_sc = float(np.mean(sc))
        
        # Decision logic
        if mean_sf > 0.7:
            return 'broadband_noise'  # High noise floor → Spectral gating
        elif harmonic_ratio > 0.7:
            return 'harmonic_rich'  # Strong harmonics → HPSS
        elif 1500 < mean_sc < 4000:
            return 'vocal_heavy'  # Mid-frequency energy → Spleeter
        else:
            return 'general'  # Everything else → Demucs ML
    
    except MemoryError:
        # Fallback: Use simpler analysis without HPSS
        print("  [Memory warning: Using simplified characterization]")
        sf = librosa.feature.spectral_flatness(y=audio[:SR*5], n_fft=n_fft, hop_length=hop_length)
        mean_sf = float(np.mean(sf))
        
        if mean_sf > 0.7:
            return 'broadband_noise'
        elif mean_sf < 0.3:
            return 'harmonic_rich'
        else:
            return 'general'
    
    except Exception as e:
        print(f"  [Warning: Characterization error: {e}, using 'general']")
        return 'general'


# ======================== FEATURE EXTRACTION ==========================
def audio_to_mel(audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, 
                 hop_length=HOP_LENGTH, target_frames=MEL_FRAMES):
    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))
    
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, 
                                          n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    if mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :target_frames]
    
    return mel_db.astype(np.float32)


def audio_to_multi_features(audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, 
                            hop_length=HOP_LENGTH, target_frames=MEL_FRAMES):
    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))
    
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, 
                                          n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    
    features = np.vstack([mel_db, chroma, contrast, mfcc])
    features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-8)
    
    if features.shape[1] < target_frames:
        pad_width = target_frames - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :target_frames]
    
    target_freq_bins = 128
    if features.shape[0] < target_freq_bins:
        pad_height = target_freq_bins - features.shape[0]
        features = np.pad(features, ((0, pad_height), (0, 0)), mode='constant')
    else:
        features = features[:target_freq_bins, :]
    
    return features.astype(np.float32)


# ======================== VISUALIZATION ================================
def save_spectrogram_comparison(audio_original, audio_processed, method_name, sample_idx=0):
    """Save before/after spectrogram comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original
    mel_orig = librosa.feature.melspectrogram(y=audio_original, sr=SR, n_mels=N_MELS)
    mel_orig_db = librosa.power_to_db(mel_orig, ref=np.max)
    img1 = axes[0].imshow(mel_orig_db, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f'Original Audio - Sample {sample_idx}')
    axes[0].set_ylabel('Mel Frequency')
    plt.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # Processed
    mel_proc = librosa.feature.melspectrogram(y=audio_processed, sr=SR, n_mels=N_MELS)
    mel_proc_db = librosa.power_to_db(mel_proc, ref=np.max)
    img2 = axes[1].imshow(mel_proc_db, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f'{method_name} - Sample {sample_idx}')
    axes[1].set_ylabel('Mel Frequency')
    axes[1].set_xlabel('Time Frames')
    plt.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    filepath = os.path.join(SPECTROGRAMS_DIR, f'{method_name}_sample{sample_idx}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f'  Saved spectrogram: {filepath}')
    return filepath


def generate_spectrogram_figure(audio_original, audio_processed, method_name, sample_idx=0):
    """Generate spectrogram comparison figure for display (returns fig object)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original
    mel_orig = librosa.feature.melspectrogram(y=audio_original, sr=SR, n_mels=N_MELS)
    mel_orig_db = librosa.power_to_db(mel_orig, ref=np.max)
    img1 = axes[0].imshow(mel_orig_db, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f'Original Audio (Baseline) - Sample {sample_idx}')
    axes[0].set_ylabel('Mel Frequency Bins')
    plt.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # Processed
    mel_proc = librosa.feature.melspectrogram(y=audio_processed, sr=SR, n_mels=N_MELS)
    mel_proc_db = librosa.power_to_db(mel_proc, ref=np.max)
    img2 = axes[1].imshow(mel_proc_db, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f'Processed Audio ({method_name}) - Sample {sample_idx}')
    axes[1].set_ylabel('Mel Frequency Bins')
    axes[1].set_xlabel('Time Frames')
    plt.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    return fig


# ======================== EVALUATION METRICS ===========================
def calculate_comprehensive_metrics(y_true, y_pred, y_pred_proba, genres):
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro-averaged metrics
    metrics['precision_macro'] = np.mean(precision)
    metrics['recall_macro'] = np.mean(recall)
    metrics['f1_macro'] = np.mean(f1)
    
    # Weighted metrics
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['precision_weighted'] = precision_w
    metrics['recall_weighted'] = recall_w
    metrics['f1_weighted'] = f1_w
    
    # AUC-ROC (one-vs-rest)
    try:
        auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        metrics['auc_roc_macro'] = auc_roc
    except:
        metrics['auc_roc_macro'] = 0.0
    
    # Per-class breakdown
    metrics['per_class'] = {
        genres[i]: {
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': support[i]
        }
        for i in range(len(genres))
    }
    
    return metrics


# ======================== PREPROCESSING PIPELINES =====================
def baseline_pipeline(audio_batch):
    X = []
    feature_func = audio_to_multi_features if USE_MULTI_FEATURES else audio_to_mel
    
    for audio in audio_batch:
        features = feature_func(audio)
        X.append(features)
    return np.array(X)[..., np.newaxis]


def assess_audio_quality(audio, sr=SR):
    # Analyze first 10 seconds (enough for assessment)
    audio_segment = audio[:sr*10]
    
    # 1. SNR Estimation (Signal-to-Noise Ratio)
    signal_power = np.mean(audio_segment**2)
    noise_floor = np.std(audio_segment[:1000])  # First 1000 samples as noise estimate
    snr_db = 10 * np.log10(signal_power / (noise_floor**2 + 1e-10))
    snr_score = min(100, max(0, (snr_db - 5) * 4))  # 5dB=0, 30dB=100
    
    # 2. Spectral Flatness (noise indicator - lower is better for music)
    sf = librosa.feature.spectral_flatness(y=audio_segment, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mean_sf = float(np.mean(sf))
    sf_score = (1 - mean_sf) * 100  # Lower flatness = more tonal = higher score
    
    # 3. Zero Crossing Rate (stability indicator)
    zcr = librosa.feature.zero_crossing_rate(audio_segment, frame_length=2048, hop_length=512)
    zcr_normalized = float(np.mean(zcr)) / 0.1  # Normalize to 0-1 range
    zcr_score = (1 - min(1, zcr_normalized)) * 100  # Lower ZCR = more stable = higher score
    
    # 4. Dynamic Range (clipping detection)
    peak_value = np.max(np.abs(audio_segment))
    clipping_detected = peak_value > 0.99
    dynamic_score = 100 if not clipping_detected else 50
    
    # 5. Spectral Rolloff (frequency content quality)
    rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    rolloff_normalized = float(np.mean(rolloff)) / (sr / 2)
    rolloff_score = rolloff_normalized * 100  # Higher rolloff = more content = better
    
    quality_score = (
        snr_score * 0.35 +
        sf_score * 0.25 +
        zcr_score * 0.15 +
        dynamic_score * 0.15 +
        rolloff_score * 0.10
    )
    
    if quality_score >= 75:
        needs_preprocessing = False
        recommendation = 'skip'
    elif quality_score >= 60:
        needs_preprocessing = True
        recommendation = 'very_light'  # 2% reduction
    elif quality_score >= 40:
        needs_preprocessing = True
        recommendation = 'light'  # 5% reduction
    else:
        needs_preprocessing = True
        recommendation = 'moderate'  # 10% reduction
    
    return quality_score, needs_preprocessing, recommendation


def adaptive_pipeline_intelligent(audio_batch):
    X = []
    feature_func = audio_to_multi_features if USE_MULTI_FEATURES else audio_to_mel
    
    method_stats = {
        'skipped_high_quality': 0,
        'very_light_denoising': 0,
        'light_denoising': 0,
        'moderate_denoising': 0
    }
    
    quality_scores = []
    
    for audio in audio_batch:
        # Assess audio quality intelligently
        quality_score, needs_preprocessing, recommendation = assess_audio_quality(audio)
        quality_scores.append(quality_score)
        
        processed_audio = audio.copy()
        
        if recommendation == 'skip':
            method_stats['skipped_high_quality'] += 1
            
        elif recommendation == 'very_light':
            processed_audio = denoise_spectral_gating(audio, threshold_db=-50, alpha=0.02)
            method_stats['very_light_denoising'] += 1
            
        elif recommendation == 'light':
            # Fair quality (40-60) → 5% noise reduction
            processed_audio = denoise_spectral_gating(audio, threshold_db=-45, alpha=0.05)
            method_stats['light_denoising'] += 1
            
        else:  # moderate
            # Poor quality (<40) → 10% noise reduction
            processed_audio = denoise_spectral_gating(audio, threshold_db=-40, alpha=0.10)
            method_stats['moderate_denoising'] += 1
        
        features = feature_func(processed_audio)
        X.append(features)
    
    # Statistics
    avg_quality = np.mean(quality_scores)
    min_quality = np.min(quality_scores)
    max_quality = np.max(quality_scores)
    
    print(f'  Intelligent quality-based preprocessing:')
    print(f'    Average quality score: {avg_quality:.1f}/100')
    print(f'    Quality range: {min_quality:.1f} - {max_quality:.1f}')
    print(f'  ')
    for method, count in method_stats.items():
        if count > 0:
            pct = (count / len(audio_batch)) * 100
            print(f'    {method:25s}: {count:4d} samples ({pct:5.1f}%)')
    
    return np.array(X)[..., np.newaxis], method_stats


def adaptive_pipeline_minimal(audio_batch):
    X = []
    feature_func = audio_to_multi_features if USE_MULTI_FEATURES else audio_to_mel
    
    method_stats = {
        'no_preprocessing': 0,
        'very_light_denoising': 0,
        'moderate_denoising': 0
    }
    
    for audio in audio_batch:
        # Quick SNR estimation (first 5 seconds only)
        signal_power = np.mean(audio[:SR*5]**2)
        noise_estimate = np.std(audio[:1000])  # First 1000 samples as noise floor
        snr_db = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10))
        
        # Spectral flatness (noise indicator)
        sf = librosa.feature.spectral_flatness(y=audio[:SR*5], n_fft=N_FFT, hop_length=HOP_LENGTH)
        mean_sf = float(np.mean(sf))
        
        processed_audio = audio.copy()
        
        # ULTRA-CONSERVATIVE: Only process very noisy samples
        if snr_db < 10 and mean_sf > 0.75:
            # VERY noisy (SNR < 10dB, high noise floor) → Moderate denoising
            # This should be <5% of FMA-medium samples
            processed_audio = denoise_spectral_gating(audio, threshold_db=-45, alpha=0.05)  # Very gentle
            method_stats['moderate_denoising'] += 1
            
        elif snr_db < 15 and mean_sf > 0.7:
            # Moderately noisy → Very light denoising (5% reduction)
            # This should be ~5-10% of samples
            processed_audio = denoise_spectral_gating(audio, threshold_db=-50, alpha=0.02)  # Extremely gentle
            method_stats['very_light_denoising'] += 1
            
        else:
            # Everything else (85-90% of samples) → NO PREPROCESSING
            # Preserve original audio to maintain genre characteristics
            method_stats['no_preprocessing'] += 1
        
        features = feature_func(processed_audio)
        X.append(features)
    
    print(f'  Minimal preprocessing applied:')
    for method, count in method_stats.items():
        if count > 0:
            pct = (count / len(audio_batch)) * 100
            print(f'    {method:25s}: {count:4d} samples ({pct:5.1f}%)')
    
    return np.array(X)[..., np.newaxis], method_stats


# ======================== MODEL-BASED ROUTING SYSTEM ====================

def extract_audio_features_for_routing(audio, sr=SR):
    """Extract features for preprocessing method routing decision"""
    try:
        # Limit to first 5 seconds for speed
        audio_segment = audio[:sr*5] if len(audio) > sr*5 else audio
        
        # SNR estimation
        signal_power = np.mean(audio_segment**2)
        noise_estimate = np.std(audio_segment[:1000])
        snr_db = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10))
        
        # Spectral features
        sf = librosa.feature.spectral_flatness(y=audio_segment, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mean_sf = float(np.mean(sf))
        std_sf = float(np.std(sf))
        
        sc = librosa.feature.spectral_centroid(y=audio_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mean_sc = float(np.mean(sc))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_segment, hop_length=HOP_LENGTH)
        mean_zcr = float(np.mean(zcr))
        
        # Harmonic-percussive characteristics
        try:
            harmonic, percussive = librosa.effects.hpss(audio_segment, margin=3.0)
            harmonic_energy = np.sum(harmonic**2)
            percussive_energy = np.sum(percussive**2)
            total_energy = np.sum(audio_segment**2) + 1e-10
            
            harmonic_ratio = harmonic_energy / total_energy
            percussive_ratio = percussive_energy / total_energy
        except:
            harmonic_ratio = 0.5
            percussive_ratio = 0.5
        
        # RMS energy
        rms = librosa.feature.rms(y=audio_segment, hop_length=HOP_LENGTH)
        mean_rms = float(np.mean(rms))
        std_rms = float(np.std(rms))
        
        # Spectral bandwidth
        bw = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mean_bw = float(np.mean(bw))
        
        features = np.array([
            snr_db,
            mean_sf,
            std_sf,
            mean_sc / sr,  # Normalize
            mean_zcr,
            harmonic_ratio,
            percussive_ratio,
            mean_rms,
            std_rms,
            mean_bw / sr  # Normalize
        ])
        
        return features
    except:
        return np.zeros(10)  # Return zero features on error


def create_routing_model(input_dim=10, n_methods=5):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(n_methods, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


ROUTING_MODEL = None
_ROUTING_FEAT_MEAN = None
_ROUTING_FEAT_STD = None
METHOD_ID_TO_NAME = {
    0: 'none',
    1: 'spectral_gating',
    2: 'hpss',
    3: 'wiener_gentle',
    4: 'spectral_subtraction'
}


def apply_preprocessing_method(audio, method_id):
    """Apply preprocessing method by ID"""
    if method_id == 0:
        return audio  # No preprocessing
    elif method_id == 1:
        return denoise_spectral_gating(audio, threshold_db=-40, alpha=0.1)
    elif method_id == 2:
        harmonic, _ = librosa.effects.hpss(audio, margin=3.0)
        return harmonic
    elif method_id == 3:
        return denoise_wiener_filter_gentle(audio)
    elif method_id == 4:
        return denoise_spectral_subtraction_gentle(audio)
    elif method_id == 5:
        return denoise_wavelet(audio)
    elif method_id == 6:
        return denoise_non_local_means(audio)
    elif method_id == 7:
        return denoise_adaptive_wiener(audio)
    elif method_id == 8:
        return denoise_multiband_spectral_subtraction(audio)
    else:
        return audio


def generate_routing_training_data(audio_batch, labels, n_samples=500, classifier_model=None, noise_snr=0):
    from tensorflow.keras.utils import to_categorical
    
    print(f"Generating training data for routing model with {min(n_samples, len(audio_batch))} samples...")
    
    # Sample subset
    n_samples_actual = min(n_samples, len(audio_batch))
    indices = np.random.choice(len(audio_batch), n_samples_actual, replace=False)
    
    X_routing = []
    y_routing = []
    
    feature_func = audio_to_multi_features if USE_MULTI_FEATURES else audio_to_mel
    n_methods = len(METHOD_ID_TO_NAME)
    
    print(f"Testing {n_methods} preprocessing methods on each sample...")
    for i, idx in enumerate(indices):
        if i % 50 == 0:
            print(f"  Progress: {i}/{n_samples_actual} samples processed...")
        
        try:
            audio = audio_batch[idx]
            
            # Add noise so the router learns meaningful distinctions across methods
            if noise_snr > 0:
                snr_db = noise_snr
            else:
                snr_db = int(np.random.choice([5, 10, 15, 20]))
            noisy_audio = _add_noise_at_snr_silent(audio, target_snr_db=snr_db)
            
            # Extract routing features from noisy audio
            routing_features = extract_audio_features_for_routing(noisy_audio)
            
            # Test each preprocessing method on the noisy audio
            method_scores = []
            for method_id in range(n_methods):
                try:
                    processed_audio = apply_preprocessing_method(noisy_audio, method_id)
                    features = feature_func(processed_audio)
                    
                    # Evaluate quality using classifier - score on true label confidence
                    if classifier_model is not None:
                        features_batch = features[np.newaxis, ..., np.newaxis]
                        pred = classifier_model.predict(features_batch, verbose=0)
                        score = float(pred[0][labels[idx]])  # Confidence on true label
                    else:
                        # Fallback: use SNR improvement as proxy
                        snr_original = 10 * np.log10(np.mean(noisy_audio**2) / (np.var(noisy_audio[:1000]) + 1e-10))
                        snr_processed = 10 * np.log10(np.mean(processed_audio**2) / (np.var(processed_audio[:1000]) + 1e-10))
                        score = max(0, (snr_processed - snr_original) / 10.0)  # Normalize
                    
                    method_scores.append(score)
                except:
                    method_scores.append(0.0)
            
            # Oracle signal: improvement over no-preprocessing baseline (method 0).
            # Defaults to 'none' when the classifier is not yet reliable.
            baseline_score = method_scores[0]
            improvements = [s - baseline_score for s in method_scores]
            best_idx = int(np.argmax(improvements))
            # Require a meaningful confidence lift. Below threshold 'none' wins
            IMPROVEMENT_THRESHOLD = 0.02
            best_method = best_idx if improvements[best_idx] >= IMPROVEMENT_THRESHOLD else 0
            
            X_routing.append(routing_features)
            y_routing.append(best_method)
            
        except Exception as e:
            continue
    
    X_routing = np.array(X_routing)
    y_routing_ids = np.array(y_routing)
    y_routing = to_categorical(y_routing_ids, num_classes=n_methods)

    # Log label distribution (heavy skew indicates unreliable oracle)
    from collections import Counter
    label_counts = Counter(y_routing_ids.tolist())
    print("Routing label distribution:")
    for method_id in sorted(label_counts):
        method_name = METHOD_ID_TO_NAME.get(method_id, f'method_{method_id}')
        cnt = label_counts[method_id]
        print(f"  {method_name:25s}: {cnt:4d} ({100*cnt/len(y_routing_ids):5.1f}%)")
    max_freq = max(label_counts.values()) / max(len(y_routing_ids), 1)
    if max_freq > 0.80:
        dominant = METHOD_ID_TO_NAME.get(int(max(label_counts, key=label_counts.get)), '?')
        print(f"⚠ WARNING: {max_freq*100:.0f}% of routing labels are '{dominant}'.")
        print(f"  Oracle accuracy may be too low for reliable routing labels.")
    
    print(f"✓ Generated {len(X_routing)} training samples for routing model")
    return X_routing, y_routing


def train_routing_model(audio_batch, labels, n_samples=500, classifier_model=None, epochs=20, noise_snr=0):
    """Train the preprocessing routing model"""
    global ROUTING_MODEL, _ROUTING_FEAT_MEAN, _ROUTING_FEAT_STD
    
    # Generate training data
    X_train, y_train = generate_routing_training_data(
        audio_batch, labels, n_samples, classifier_model, noise_snr=noise_snr
    )
    
    if len(X_train) == 0:
        print("ERROR: No training data generated. Cannot train routing model.")
        return None, 0.0
    
    # Z-score normalize routing features (scales differ significantly across features)
    _ROUTING_FEAT_MEAN = np.mean(X_train, axis=0)
    _ROUTING_FEAT_STD  = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - _ROUTING_FEAT_MEAN) / _ROUTING_FEAT_STD
    
    # Balanced class weights to counteract routing label imbalance
    y_ids = np.argmax(y_train, axis=1)
    from sklearn.utils.class_weight import compute_class_weight
    classes_present = np.unique(y_ids)
    cw = compute_class_weight('balanced', classes=classes_present, y=y_ids)
    class_weights = {int(c): float(w) for c, w in zip(classes_present, cw)}
    for c in range(y_train.shape[1]):
        if c not in class_weights:
            class_weights[c] = 1.0

    ROUTING_MODEL = create_routing_model(input_dim=X_train.shape[1], n_methods=y_train.shape[1])

    print("\nTraining routing neural network...")
    history = ROUTING_MODEL.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        verbose=1
    )
    
    final_accuracy = history.history['accuracy'][-1]
    print(f"\n✓ Routing model training complete. Final accuracy: {final_accuracy:.3f}")
    
    return ROUTING_MODEL, final_accuracy


def adaptive_pipeline_model_based(audio_batch, return_decisions=False):
    global ROUTING_MODEL
    
    X = []
    feature_func = audio_to_multi_features if USE_MULTI_FEATURES else audio_to_mel
    
    method_stats = {name: 0 for name in METHOD_ID_TO_NAME.values()}
    routing_decisions_log = []  # filled when return_decisions=True
    
    # Fallback to granular if model not trained
    if ROUTING_MODEL is None:
        print("  WARNING: Routing model not trained. Using rule-based fallback.")
        result = adaptive_pipeline_granular(audio_batch)
        if return_decisions:
            return result[0], result[1], []
        return result
    
    for audio in audio_batch:
        try:
            # Extract routing features
            routing_features = extract_audio_features_for_routing(audio)
            
            # Apply same standardization used during training
            if _ROUTING_FEAT_MEAN is not None and _ROUTING_FEAT_STD is not None:
                routing_features = (routing_features - _ROUTING_FEAT_MEAN) / _ROUTING_FEAT_STD
            
            # Predict best preprocessing method
            routing_features_batch = routing_features[np.newaxis, :]
            predictions = ROUTING_MODEL.predict(routing_features_batch, verbose=0)
            method_id = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][method_id])
            
            # Apply preprocessing (with confidence threshold)
            if confidence > 0.25:  # Use model prediction if confident
                processed_audio = apply_preprocessing_method(audio, method_id)
                method_name = METHOD_ID_TO_NAME[method_id]
            else:  # Fallback to no preprocessing if uncertain
                processed_audio = audio
                method_name = 'none'
                method_id = 0
                confidence = float(predictions[0][0])
            
            method_stats[method_name] += 1
            
            if return_decisions:
                routing_decisions_log.append({
                    'method': method_name,
                    'method_id': method_id,
                    'confidence': round(confidence, 4),
                    'all_probabilities': {
                        METHOD_ID_TO_NAME[i]: round(float(predictions[0][i]), 4)
                        for i in range(len(predictions[0]))
                    },
                })
            
            # Extract features
            features = feature_func(processed_audio)
            X.append(features)
            
        except Exception as e:
            # On error, use original audio
            features = feature_func(audio)
            X.append(features)
            method_stats['none'] += 1
            if return_decisions:
                routing_decisions_log.append({'method': 'none', 'method_id': 0,
                                              'confidence': 0.0, 'all_probabilities': {}})
    
    print(f'  Model-based preprocessing applied:')
    for method, count in method_stats.items():
        if count > 0:
            pct = (count / len(audio_batch)) * 100
            print(f'    {method:25s}: {count:4d} samples ({pct:5.1f}%)')
    
    X_arr = np.array(X)[..., np.newaxis]
    if return_decisions:
        return X_arr, method_stats, routing_decisions_log
    return X_arr, method_stats


# ======================== END MODEL-BASED ROUTING ====================


def adaptive_pipeline_granular(audio_batch):
    X = []
    feature_func = audio_to_multi_features if USE_MULTI_FEATURES else audio_to_mel
    
    method_stats = {
        'no_preprocessing': 0,
        'spectral_gating_only': 0,
        'hpss_only': 0,
        'hpss_and_gating': 0,
        'wiener_only': 0,
        'spectral_subtraction': 0
    }
    
    for audio in audio_batch:
        # Analyze audio characteristics
        sf = librosa.feature.spectral_flatness(y=audio[:SR*5], n_fft=N_FFT, hop_length=HOP_LENGTH)
        mean_sf = float(np.mean(sf))
        
        # Estimate SNR
        signal_power = np.mean(audio**2)
        noise_estimate = np.std(audio[:1000])
        snr_db = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10))
        
        # Quick harmonic ratio
        try:
            harmonic, percussive = librosa.effects.hpss(audio[:SR*5], margin=3.0)
            harmonic_energy = np.sum(harmonic**2)
            total_energy = np.sum(audio[:SR*5]**2)
            harmonic_ratio = harmonic_energy / (total_energy + 1e-8)
        except:
            harmonic_ratio = 0.5  # Neutral if error
        
        processed_audio = audio.copy()
        
        if snr_db > 25 and mean_sf < 0.4:
            method_stats['no_preprocessing'] += 1
            
        elif mean_sf > 0.7 and snr_db < 15:
            processed_audio = denoise_spectral_subtraction_gentle(audio)
            method_stats['spectral_subtraction'] += 1
            
        elif mean_sf > 0.6:
            processed_audio = denoise_spectral_gating(audio, threshold_db=-40, alpha=0.1)
            method_stats['spectral_gating_only'] += 1
            
        elif harmonic_ratio > 0.75 and mean_sf < 0.4:
            harmonic, _ = librosa.effects.hpss(audio, margin=3.0)
            processed_audio = harmonic
            method_stats['hpss_only'] += 1
            
        elif harmonic_ratio > 0.65 and mean_sf > 0.5:
            # Harmonics + noise = HPSS then gating
            harmonic, _ = librosa.effects.hpss(audio, margin=3.0)
            processed_audio = denoise_spectral_gating(harmonic, threshold_db=-35, alpha=0.1)
            method_stats['hpss_and_gating'] += 1
            
        else:
            # General case = Gentle Wiener filter (safe, minimal)
            processed_audio = denoise_wiener_filter_gentle(audio)
            method_stats['wiener_only'] += 1
        
        features = feature_func(processed_audio)
        X.append(features)
    
    print(f'  Granular preprocessing applied:')
    for method, count in method_stats.items():
        if count > 0:
            pct = (count / len(audio_batch)) * 100
            print(f'    {method:25s}: {count:4d} samples ({pct:5.1f}%)')
    
    return np.array(X)[..., np.newaxis], method_stats


def adaptive_pipeline_hpss(audio_batch):
    X = []
    feature_func = audio_to_multi_features if USE_MULTI_FEATURES else audio_to_mel
    
    routing_stats = {
        'broadband_noise': 0,
        'harmonic_rich': 0,
        'vocal_heavy': 0,
        'general': 0
    }
    
    for audio in audio_batch:
        # Characterize audio
        audio_type = characterize_audio_type(audio)
        routing_stats[audio_type] += 1
        
        # Route to appropriate method (aggressive preprocessing)
        if audio_type == 'broadband_noise':
            audio = denoise_spectral_gating(audio, threshold_db=-40, alpha=0.1)
        elif audio_type == 'harmonic_rich':
            harmonic, percussive = librosa.effects.hpss(audio, margin=3.0)
            audio = harmonic  # Keep only harmonic (removes percussion!)
        elif audio_type == 'vocal_heavy':
            audio = denoise_spectral_gating(audio, threshold_db=-35, alpha=0.15)
        else:  # general
            audio = denoise_spectral_gating(audio, threshold_db=-40, alpha=0.1)
        
        features = feature_func(audio)
        X.append(features)
    
    print(f'  HPSS Pipeline routing: Broadband={routing_stats["broadband_noise"]}, '
          f'Harmonic={routing_stats["harmonic_rich"]}, '
          f'Vocal={routing_stats["vocal_heavy"]}, '
          f'General={routing_stats["general"]}')
    
    return np.array(X)[..., np.newaxis], routing_stats


# Legacy alias for backward compatibility
adaptive_pipeline = adaptive_pipeline_hpss


# ==================== GENTLE PREPROCESSING ====================
def denoise_spectral_subtraction_gentle(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Gentle spectral subtraction"""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    
    noise_profile = np.median(mag[:, :10], axis=1, keepdims=True)
    mag_clean = np.maximum(mag - 0.5 * noise_profile, 0.0)  # 50% noise reduction
    
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    return audio_clean


def denoise_median_filtering_gentle(audio, kernel_size=5):
    """Gentle median filtering"""
    # Convert to mel-spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Apply gentle median filter
    mel_filtered = medfilt2d(mel_db, kernel_size=kernel_size)
    
    # Blend 60% filtered + 40% original (preserve musical content)
    mel_blended = 0.6 * mel_filtered + 0.4 * mel_db
    
    # Convert back (approximate)
    mel_power = librosa.db_to_power(mel_blended)
    audio_clean = librosa.feature.inverse.mel_to_audio(mel_power, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    if len(audio_clean) > len(audio):
        audio_clean = audio_clean[:len(audio)]
    elif len(audio_clean) < len(audio):
        audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
    
    return audio_clean


def denoise_wiener_filter_gentle(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Gentle Wiener filter"""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    
    noise_profile = np.median(mag[:, :10], axis=1, keepdims=True)
    signal_power = mag ** 2
    noise_power = noise_profile ** 2
    
    # Gentle Wiener gain with floor at 0.5 (preserve 50% minimum)
    wiener_gain = signal_power / (signal_power + noise_power + 1e-8)
    wiener_gain = np.maximum(wiener_gain, 0.5)
    
    mag_clean = mag * wiener_gain
    
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    return audio_clean


# ======================== ADVANCED PREPROCESSING TECHNIQUES ============

def denoise_wavelet(audio, wavelet='db4', level=4, threshold_scale=0.5):
    if not PYWT_AVAILABLE:
        return audio  # Fallback to original if pywt not available
    
    try:
        # Decompose signal
        coeffs = pywt.wavedec(audio, wavelet, level=level)
        
        # Calculate threshold using MAD (Median Absolute Deviation)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(audio))) * threshold_scale
        
        # Apply soft thresholding to detail coefficients
        coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
        for detail_coeff in coeffs[1:]:
            coeffs_thresh.append(pywt.threshold(detail_coeff, threshold, mode='soft'))
        
        # Reconstruct signal
        audio_clean = pywt.waverec(coeffs_thresh, wavelet)
        
        # Match length
        if len(audio_clean) > len(audio):
            audio_clean = audio_clean[:len(audio)]
        elif len(audio_clean) < len(audio):
            audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))
        
        return audio_clean
    except:
        return audio  # Return original on error


def denoise_non_local_means(audio, patch_size=512, search_window=2048, h=0.1):
    try:
        audio_clean = np.zeros_like(audio)
        weights_sum = np.zeros_like(audio)
        
        n = len(audio)
        half_patch = patch_size // 2
        half_search = search_window // 2
        
        # Process each position (downsampled for efficiency)
        stride = patch_size // 4
        for i in range(half_patch, n - half_patch, stride):
            # Extract reference patch
            ref_patch = audio[i - half_patch:i + half_patch]
            
            # Search window bounds
            search_start = max(half_patch, i - half_search)
            search_end = min(n - half_patch, i + half_search)
            
            # Compare with patches in search window
            for j in range(search_start, search_end, stride):
                # Extract comparison patch
                comp_patch = audio[j - half_patch:j + half_patch]
                
                # Calculate patch distance
                distance = np.sum((ref_patch - comp_patch) ** 2) / patch_size
                
                # Calculate weight using Gaussian kernel
                weight = np.exp(-distance / (h ** 2))
                
                # Accumulate weighted patches
                audio_clean[i] += weight * audio[j]
                weights_sum[i] += weight
        
        # Normalize
        mask = weights_sum > 0
        audio_clean[mask] /= weights_sum[mask]
        
        # Fill gaps with original values
        audio_clean[~mask] = audio[~mask]
        
        return audio_clean
    except:
        return audio  # Return original on error


def denoise_adaptive_wiener(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    try:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(stft), np.angle(stft)
        
        # Adaptive noise estimation using minimum statistics
        noise_profile = np.ones_like(mag) * np.inf
        window_size = 10
        
        for i in range(mag.shape[1]):
            start = max(0, i - window_size)
            end = min(mag.shape[1], i + window_size)
            noise_profile[:, i] = np.min(mag[:, start:end], axis=1)
        
        # Time-varying signal and noise power
        signal_power = mag ** 2
        noise_power = noise_profile ** 2
        
        # Adaptive Wiener gain with frequency-dependent floor
        freq_bins = np.arange(mag.shape[0]) / mag.shape[0]
        min_gain = 0.3 + 0.4 * freq_bins.reshape(-1, 1)  # Higher floor for low freq
        
        wiener_gain = signal_power / (signal_power + noise_power + 1e-8)
        wiener_gain = np.maximum(wiener_gain, min_gain)
        
        # Apply gain
        mag_clean = mag * wiener_gain
        
        # Reconstruct
        stft_clean = mag_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
        
        return audio_clean
    except:
        return audio


def denoise_multiband_spectral_subtraction(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=4):
    try:
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        mag, phase = np.abs(stft), np.angle(stft)
        
        # Estimate noise from initial frames
        noise_profile = np.mean(mag[:, :10], axis=1)
        
        # Divide spectrum into bands
        n_freq_bins = mag.shape[0]
        band_size = n_freq_bins // n_bands
        
        mag_clean = np.zeros_like(mag)
        
        for band_idx in range(n_bands):
            start_bin = band_idx * band_size
            end_bin = (band_idx + 1) * band_size if band_idx < n_bands - 1 else n_freq_bins
            
            # Band-specific parameters (more conservative for low frequencies)
            alpha = 2.0 - 0.3 * band_idx  # Oversubtraction factor
            beta = 0.01 * (band_idx + 1)  # Spectral floor
            
            # Extract band
            mag_band = mag[start_bin:end_bin, :]
            noise_band = noise_profile[start_bin:end_bin].reshape(-1, 1)
            
            # Spectral subtraction
            mag_band_clean = mag_band - alpha * noise_band
            mag_band_clean = np.maximum(mag_band_clean, beta * mag_band)
            
            mag_clean[start_bin:end_bin, :] = mag_band_clean
        
        # Reconstruct
        stft_clean = mag_clean * np.exp(1j * phase)
        audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
        
        return audio_clean
    except:
        return audio


# ======================== END ADVANCED PREPROCESSING ====================


def characterize_noise_type_gentle(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Conservative noise characterization"""
    sf = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    mean_sf = float(np.mean(sf))
    
    sc = librosa.feature.spectral_centroid(y=audio, sr=SR, n_fft=n_fft, hop_length=hop_length)
    mean_sc = float(np.mean(sc))
    
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)
    mean_zcr = float(np.mean(zcr))
    
    # Conservative thresholds
    if mean_sf > 0.75:
        return 'broadband'
    elif mean_sc < 1500:
        return 'lowfreq'
    elif mean_zcr > 0.20:
        return 'transient'
    else:
        return 'general'


def adaptive_pipeline_gentle(audio_batch):
    X = []
    feature_func = audio_to_multi_features if USE_MULTI_FEATURES else audio_to_mel
    noise_stats = {'broadband': 0, 'lowfreq': 0, 'transient': 0, 'general': 0}
    
    for audio in audio_batch:
        noise_type = characterize_noise_type_gentle(audio)
        noise_stats[noise_type] += 1
        
        if noise_type == 'broadband':
            audio_clean = denoise_spectral_subtraction_gentle(audio)
        elif noise_type in ['lowfreq', 'transient']:
            audio_clean = denoise_median_filtering_gentle(audio)
        else:  # general
            audio_clean = denoise_wiener_filter_gentle(audio)
        
        features = feature_func(audio_clean)
        X.append(features)
    
    print(f'  Gentle preprocessing: Broadband={noise_stats["broadband"]}, '
          f'LowFreq={noise_stats["lowfreq"]}, Transient={noise_stats["transient"]}, '
          f'General={noise_stats["general"]}')
    
    return np.array(X)[..., np.newaxis], noise_stats


# ==================== DATASET-LEVEL NOISE ANALYSIS ====================
def analyze_dataset_noise(audio_batch, sample_size=8000):
    print(f'\nAnalyzing dataset characteristics ({sample_size} samples)...')
    
    # Limit to sample_size
    sample_audio = audio_batch[:sample_size] if len(audio_batch) > sample_size else audio_batch
    
    # Metrics for GRANULAR method
    snr_values = []
    spectral_flatness_values = []
    harmonic_ratio_values = []
    
    # Metrics for INTELLIGENT method
    quality_scores = []
    
    # Predicted routing for GRANULAR method
    granular_routing = {
        'no_preprocessing': 0,
        'spectral_gating_only': 0,
        'hpss_only': 0,
        'hpss_and_gating': 0,
        'wiener_only': 0,
        'spectral_subtraction': 0
    }
    
    # Predicted routing for INTELLIGENT method
    intelligent_routing = {
        'skipped_high_quality': 0,
        'very_light_denoising': 0,
        'light_denoising': 0,
        'moderate_denoising': 0
    }
    
    print('Analyzing audio characteristics...')
    for i, audio in enumerate(sample_audio):
        try:
            analysis_length = min(len(audio), SR*5)
            
            sf = librosa.feature.spectral_flatness(y=audio[:analysis_length], n_fft=N_FFT, hop_length=HOP_LENGTH)
            mean_sf = float(np.mean(sf))
            spectral_flatness_values.append(mean_sf)
            
            signal_power = np.mean(audio**2)
            noise_sample_size = min(1000, len(audio) // 2)
            noise_estimate = np.std(audio[:noise_sample_size])
            
            if signal_power < 1e-10 or noise_estimate < 1e-10:
                snr_db = 20.0
            else:
                snr_db = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10))
                snr_db = np.clip(snr_db, -20, 60)
            
            snr_values.append(snr_db)
            
            # 3. Harmonic ratio
            try:
                harmonic, percussive = librosa.effects.hpss(audio[:analysis_length], margin=3.0)
                harmonic_energy = np.sum(harmonic**2)
                total_energy = np.sum(audio[:analysis_length]**2)
                harmonic_ratio = harmonic_energy / (total_energy + 1e-8)
            except:
                harmonic_ratio = 0.5
            harmonic_ratio_values.append(harmonic_ratio)
        except Exception as e:
            # Skip problematic samples, use neutral values
            spectral_flatness_values.append(0.5)
            snr_values.append(20.0)
            harmonic_ratio_values.append(0.5)
            mean_sf = 0.5
            snr_db = 20.0
            harmonic_ratio = 0.5
            print(f'  Warning: Skipped sample {i} due to error: {str(e)[:50]}')
        
        # Predict GRANULAR routing (match granular pipeline logic)
        if snr_db > 25 and mean_sf < 0.4:
            granular_routing['no_preprocessing'] += 1
        elif mean_sf > 0.7 and snr_db < 15:
            granular_routing['spectral_subtraction'] += 1
        elif mean_sf > 0.6:
            granular_routing['spectral_gating_only'] += 1
        elif harmonic_ratio > 0.75 and mean_sf < 0.4:
            granular_routing['hpss_only'] += 1
        elif harmonic_ratio > 0.65 and mean_sf > 0.5:
            granular_routing['hpss_and_gating'] += 1
        else:
            granular_routing['wiener_only'] += 1
        
        # === INTELLIGENT METHOD ANALYSIS ===
        try:
            quality_score, needs_preprocessing, recommendation = assess_audio_quality(audio)
            quality_scores.append(quality_score)
            
            # Predict INTELLIGENT routing
            if recommendation == 'skip':
                intelligent_routing['skipped_high_quality'] += 1
            elif recommendation == 'very_light':
                intelligent_routing['very_light_denoising'] += 1
            elif recommendation == 'light':
                intelligent_routing['light_denoising'] += 1
            else:
                intelligent_routing['moderate_denoising'] += 1
        except Exception as e:
            # Default to moderate quality if error
            quality_scores.append(50.0)
            intelligent_routing['light_denoising'] += 1
        
        if (i + 1) % 100 == 0:
            print(f'  Analyzed {i + 1}/{len(sample_audio)} samples...')
    
    total = len(sample_audio)
    
    print('\n' + '='*70)
    print('DATASET ANALYSIS FOR GRANULAR & INTELLIGENT METHODS')
    print('='*70)
    
    # GRANULAR METHOD ANALYSIS
    print('\n[GRANULAR METHOD] Audio Quality Metrics:')
    print(f'  SNR (Signal-to-Noise Ratio):')
    print(f'    Mean: {np.mean(snr_values):6.2f} dB  |  Median: {np.median(snr_values):6.2f} dB')
    print(f'    Range: {np.min(snr_values):6.2f} to {np.max(snr_values):6.2f} dB')
    
    print(f'  Spectral Flatness (0=tonal, 1=noisy):')
    print(f'    Mean: {np.mean(spectral_flatness_values):.3f}  |  Median: {np.median(spectral_flatness_values):.3f}')
    print(f'    Range: {np.min(spectral_flatness_values):.3f} to {np.max(spectral_flatness_values):.3f}')
    
    print(f'  Harmonic Ratio (0=noise, 1=pure harmonic):')
    print(f'    Mean: {np.mean(harmonic_ratio_values):.3f}  |  Median: {np.median(harmonic_ratio_values):.3f}')
    print(f'    Range: {np.min(harmonic_ratio_values):.3f} to {np.max(harmonic_ratio_values):.3f}')
    
    print('\n[GRANULAR METHOD] Predicted Preprocessing Routes:')
    for method, count in granular_routing.items():
        pct = (count / total) * 100
        bar = '█' * int(pct / 2)
        print(f'  {method:25s}: {count:4d} samples ({pct:5.1f}%) {bar}')
    
    # INTELLIGENT METHOD ANALYSIS
    print('\n[INTELLIGENT METHOD] Quality Score Distribution:')
    print(f'  Mean Quality: {np.mean(quality_scores):6.2f}/100  |  Median: {np.median(quality_scores):6.2f}/100')
    print(f'  Range: {np.min(quality_scores):6.2f} to {np.max(quality_scores):6.2f}')
    
    # Quality distribution bins
    excellent = sum(1 for q in quality_scores if q >= 75)
    good = sum(1 for q in quality_scores if 60 <= q < 75)
    fair = sum(1 for q in quality_scores if 40 <= q < 60)
    poor = sum(1 for q in quality_scores if q < 40)
    
    print(f'  Excellent (≥75): {excellent:4d} ({excellent/total*100:5.1f}%)')
    print(f'  Good (60-75):    {good:4d} ({good/total*100:5.1f}%)')
    print(f'  Fair (40-60):    {fair:4d} ({fair/total*100:5.1f}%)')
    print(f'  Poor (<40):      {poor:4d} ({poor/total*100:5.1f}%)')
    
    print('\n[INTELLIGENT METHOD] Predicted Preprocessing Routes:')
    for method, count in intelligent_routing.items():
        pct = (count / total) * 100
        bar = '█' * int(pct / 2)
        print(f'  {method:25s}: {count:4d} samples ({pct:5.1f}%) {bar}')
    
    print('\n' + '='*70)
    
    return {
        'total_analyzed': total,
        'granular_metrics': {
            'snr_mean': np.mean(snr_values),
            'snr_median': np.median(snr_values),
            'sf_mean': np.mean(spectral_flatness_values),
            'sf_median': np.median(spectral_flatness_values),
            'hr_mean': np.mean(harmonic_ratio_values),
            'hr_median': np.median(harmonic_ratio_values),
            'routing': granular_routing
        },
        'intelligent_metrics': {
            'quality_mean': np.mean(quality_scores),
            'quality_median': np.median(quality_scores),
            'excellent_pct': excellent/total*100,
            'good_pct': good/total*100,
            'fair_pct': fair/total*100,
            'poor_pct': poor/total*100,
            'routing': intelligent_routing
        }
    }


def calculate_preprocessing_scores(dataset_analysis):
    granular_metrics = dataset_analysis['granular_metrics']
    intelligent_metrics = dataset_analysis['intelligent_metrics']
    
    # Score GRANULAR method (best for diverse/noisy datasets)
    # Higher score when: varied SNR, varied spectral flatness, needs different preprocessing per sample
    snr_mean = granular_metrics['snr_mean']
    sf_mean = granular_metrics['sf_mean']
    routing = granular_metrics['routing']
    total = dataset_analysis['total_analyzed']
    
    # How diverse is the preprocessing needed?
    preprocessing_diversity = sum(1 for count in routing.values() if count > total * 0.05)
    no_preprocessing_pct = routing['no_preprocessing'] / total * 100
    
    granular_score = 0
    granular_score += (100 - no_preprocessing_pct) * 0.4  # Higher if samples need preprocessing
    granular_score += preprocessing_diversity * 10         # Higher if diverse preprocessing needed
    granular_score += (sf_mean * 100) * 0.3               # Higher if noisy (high spectral flatness)
    granular_score += max(0, (25 - snr_mean)) * 2         # Higher if low SNR
    
    
    quality_mean = intelligent_metrics['quality_mean']
    excellent_pct = intelligent_metrics['excellent_pct']
    skip_pct = intelligent_metrics['routing']['skipped_high_quality'] / total * 100
    
    intelligent_score = 0
    intelligent_score += quality_mean * 0.5               # Higher if high average quality
    intelligent_score += excellent_pct * 0.8              # Higher if many excellent samples
    intelligent_score += skip_pct * 0.6                   # Higher if most can skip preprocessing
    
    # Recommendation
    if intelligent_score > granular_score:
        recommendation = 'intelligent'
        confidence = (intelligent_score - granular_score) / max(intelligent_score, granular_score)
    else:
        recommendation = 'granular'
        confidence = (granular_score - intelligent_score) / max(intelligent_score, granular_score)
    
    # Override if scores are close (within 10 points)
    if abs(granular_score - intelligent_score) < 10:
        recommendation = 'granular'
        confidence = 0.5
    
    return {
        'granular_score': granular_score,
        'intelligent_score': intelligent_score,
        'recommendation': recommendation,
        'confidence': confidence,
        'reasoning': {
            'snr_mean': snr_mean,
            'quality_mean': quality_mean,
            'preprocessing_diversity': preprocessing_diversity,
            'no_preprocessing_pct': no_preprocessing_pct,
            'excellent_quality_pct': excellent_pct
        }
    }


# ======================== MODEL ARCHITECTURE ==========================
def build_genre_cnn(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.GaussianNoise(0.15),  # SpecAugment-like regularisation

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


def get_training_callbacks(patience=7):
    """Standard callbacks for CNN training: LR scheduling + early stopping."""
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0
        ),
    ]


# ======================== SILENT NOISE ADDITION ======================
def _generate_realistic_noise_silent(duration, sr=SR):
    samples = int(duration * sr)
    
    # Pink noise base
    white = np.random.randn(samples)
    freqs = np.fft.rfftfreq(samples, 1/sr)
    freqs[0] = 1
    pink_filter = 1 / np.sqrt(freqs)
    pink_fft = np.fft.rfft(white) * pink_filter
    pink = np.fft.irfft(pink_fft, n=samples)
    pink = pink / np.std(pink) * 0.6
    
    # High-frequency hiss
    hiss = np.random.normal(0, 1, samples)
    b, a = signal.butter(2, 4000 / (sr/2), btype='high')
    hiss = signal.filtfilt(b, a, hiss) * 0.3
    
    # Low-frequency rumble
    rumble_white = np.random.randn(samples)
    rumble_freqs = np.fft.rfftfreq(samples, 1/sr)
    rumble_freqs[0] = 1
    rumble_filter = 1 / rumble_freqs
    rumble_fft = np.fft.rfft(rumble_white) * rumble_filter
    rumble = np.fft.irfft(rumble_fft, n=samples)
    b, a = signal.butter(2, 200 / (sr/2), btype='low')
    rumble = signal.filtfilt(b, a, rumble) * 0.4
    
    # Combine
    noise = pink + hiss + rumble
    noise = noise / np.std(noise)
    return noise


def _add_noise_at_snr_silent(audio, target_snr_db=15):
    signal_power = np.mean(audio ** 2)
    
    if signal_power < 1e-10:
        return audio
    
    duration = len(audio) / SR
    noise = _generate_realistic_noise_silent(duration, SR)
    
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:
        return audio
    
    target_noise_power = signal_power / (10 ** (target_snr_db / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / (noise_power + 1e-10))
    
    # Add noise
    noisy_audio = audio + noise_scaled
    
    # Prevent clipping
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 0.99:
        noisy_audio = noisy_audio * 0.99 / max_val
    
    return noisy_audio


# ======================== DATASET LOADING =============================
def load_fma_dataset(max_samples=None):
    print('Loading FMA metadata...')
    tracks = pd.read_csv(FMA_METADATA, index_col=0, header=[0, 1])
    genre_col = ('track', 'genre_top')

    # ---- Build per-genre file lists from metadata first ----------------
    print('Scanning audio files by genre...')
    all_audio_files = sorted(glob.glob(os.path.join(FMA_AUDIO_DIR, '*', '*.mp3')))

    # Map each file to its genre (skip unknowns early)
    genre_files = defaultdict(list)
    for filepath in all_audio_files:
        try:
            track_id = int(os.path.splitext(os.path.basename(filepath))[0])
            if track_id not in tracks.index:
                continue
            genre = tracks.loc[track_id, genre_col]
            if pd.isna(genre) or genre not in TARGET_GENRES:
                continue
            genre_files[genre].append(filepath)
        except Exception:
            continue

    # ---- Stratified cap: equal samples per genre -----------------------
    if max_samples:
        per_genre_limit = max_samples // len(TARGET_GENRES)
    else:
        per_genre_limit = min(len(v) for v in genre_files.values() if v)

    print(f'\nClass distribution (capped at {per_genre_limit} per genre):')
    selected_files = []  # list of (filepath, genre)
    for genre in sorted(genre_files.keys()):
        files = genre_files[genre][:per_genre_limit]
        print(f'  {genre:<15}: {len(genre_files[genre]):5d} available  →  {len(files):4d} selected')
        for fp in files:
            selected_files.append((fp, genre))

    print(f'\nLoading {len(selected_files)} audio files (stratified)...')
    X, y = [], []
    skipped = 0

    for i, (filepath, genre) in enumerate(selected_files):
        try:
            audio, sr = librosa.load(filepath, sr=SR, duration=DURATION, mono=True)

            if len(audio) < SR * 5:
                skipped += 1
                continue

            signal_power = np.mean(audio ** 2)
            if signal_power < 1e-8:
                skipped += 1
                continue

            X.append(audio)
            y.append(genre)

            if (i + 1) % 100 == 0:
                print(f'  Processed {i + 1}/{len(selected_files)} files, loaded {len(X)} valid tracks...')

        except KeyboardInterrupt:
            raise
        except Exception:
            skipped += 1
            continue

    print(f'Loaded {len(X)} tracks, skipped {skipped} files')

    # Print final class distribution
    from collections import Counter
    final_dist = Counter(y)
    total = len(y)
    print('\nFinal class distribution:')
    for g in sorted(final_dist):
        pct = final_dist[g] / total * 100
        print(f'  {g:<15}: {final_dist[g]:4d} ({pct:5.1f}%)')

    genre_to_idx = {g: i for i, g in enumerate(sorted(set(y)))}
    y_idx = np.array([genre_to_idx[g] for g in y], dtype=np.int32)

    return X, y_idx, list(genre_to_idx.keys())


# ======================== AUDIOSET DATASET LOADING ====================
def load_audioset_music_dataset(max_samples=None):
    AUDIOSET_CLIP_DURATION = 10  # AudioSet segments are 10 seconds

    if not os.path.isdir(AUDIOSET_MUSIC_DIR):
        raise FileNotFoundError(
            f'AudioSet music directory not found: {AUDIOSET_MUSIC_DIR}\n'
            f'Run:  python scripts/download_audioset_music.py --max-per-genre 500'
        )

    genre_dirs = sorted([
        d for d in os.listdir(AUDIOSET_MUSIC_DIR)
        if os.path.isdir(os.path.join(AUDIOSET_MUSIC_DIR, d))
        and d in AUDIOSET_TARGET_GENRES
    ])

    if not genre_dirs:
        raise FileNotFoundError(
            f'No genre subdirectories found under {AUDIOSET_MUSIC_DIR}.\n'
            f'Run:  python scripts/download_audioset_music.py --max-per-genre 500'
        )

    print(f'Loading AudioSet music dataset from {AUDIOSET_MUSIC_DIR} ...')
    print(f'  Found genres: {genre_dirs}')

    # Collect all WAV files across genres
    all_files = []
    for genre in genre_dirs:
        genre_dir = os.path.join(AUDIOSET_MUSIC_DIR, genre)
        wavs = glob.glob(os.path.join(genre_dir, '*.wav'))
        for w in wavs:
            all_files.append((w, genre))

    if max_samples and len(all_files) > max_samples:
        # Stratified sub-sample to keep genre balance
        import random
        random.shuffle(all_files)
        # Keep proportional split
        per_genre_limit = max_samples // len(genre_dirs)
        genre_counts = defaultdict(int)
        filtered = []
        for fp, genre in all_files:
            if genre_counts[genre] < per_genre_limit:
                filtered.append((fp, genre))
                genre_counts[genre] += 1
        all_files = filtered

    print(f'  Loading {len(all_files)} audio files...')

    X, y = [], []
    skipped = 0

    for i, (filepath, genre) in enumerate(all_files):
        try:
            audio, sr = librosa.load(
                filepath,
                sr=SR,
                duration=AUDIOSET_CLIP_DURATION,
                mono=True
            )

            if len(audio) < SR * 3:   # need at least 3 seconds
                skipped += 1
                continue

            signal_power = np.mean(audio ** 2)
            if signal_power < 1e-8:   # silence guard
                skipped += 1
                continue

            X.append(audio)
            y.append(genre)

            if (i + 1) % 100 == 0:
                print(f'  Processed {i + 1}/{len(all_files)} files, loaded {len(X)} valid tracks...')

        except KeyboardInterrupt:
            raise
        except Exception:
            skipped += 1
            continue

    print(f'Loaded {len(X)} AudioSet tracks, skipped {skipped} files')

    # Print class distribution
    from collections import Counter
    dist = Counter(y)
    for g in sorted(dist):
        print(f'  {g:<15}: {dist[g]:>4} clips')

    genre_to_idx = {g: i for i, g in enumerate(sorted(set(y)))}
    y_idx = np.array([genre_to_idx[g] for g in y], dtype=np.int32)

    return X, y_idx, list(genre_to_idx.keys())


# ======================== EXPERIMENT ==================================
def run_comparative_experiment(use_model_based_routing=False, train_routing_model_flag=False):
    print('='*70)
    print('ADAPTIVE PREPROCESSING FOR MUSIC GENRE CLASSIFICATION')
    print('='*70)
    
    # ========== STEP 1: LOAD DATASET ==========
    print('\n' + '='*70)
    print('STEP 1: LOAD DATASET')
    print('='*70)
    
    X_audio, y, genres = load_fma_dataset(max_samples=MAX_SAMPLES)
    print(f'\nGenres: {genres}')
    print(f'Total samples: {len(X_audio)}')
    
    # Train/test split
    X_train_audio, X_test_audio, y_train, y_test = train_test_split(
        X_audio, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f'Train: {len(X_train_audio)}, Test: {len(X_test_audio)}')
    
    num_classes = len(genres)
    
    # ========== STEP 2: TRAIN BASELINE MODEL ==========
    print('\n' + '='*70)
    print('STEP 2: TRAIN BASELINE MODEL (No Preprocessing)')
    print('='*70)
    
    print('Generating baseline features...')
    X_train_baseline = baseline_pipeline(X_train_audio)
    X_test_baseline = baseline_pipeline(X_test_audio)
    
    input_shape = X_train_baseline.shape[1:]
    print(f'Input shape: {input_shape}')
    
    print('\nTraining baseline CNN...')
    from sklearn.utils.class_weight import compute_class_weight
    cw_values = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw_values))
    baseline_model = build_genre_cnn(input_shape, num_classes)
    baseline_model.fit(
        X_train_baseline, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        verbose=1
    )
    
    print('\nEvaluating baseline model...')
    y_pred_proba_baseline = baseline_model.predict(X_test_baseline, verbose=0)
    preds_baseline = np.argmax(y_pred_proba_baseline, axis=1)
    
    baseline_metrics = calculate_comprehensive_metrics(y_test, preds_baseline, y_pred_proba_baseline, genres)
    
    print(f'\nBaseline Results:')
    print(f'  Accuracy:     {baseline_metrics["accuracy"]:.4f} ({baseline_metrics["accuracy"]*100:.2f}%)')
    print(f'  F1-Score:     {baseline_metrics["f1_macro"]:.4f}')
    print(f'  AUC-ROC:      {baseline_metrics["auc_roc_macro"]:.4f}')
    print(f'  Precision:    {baseline_metrics["precision_weighted"]:.4f}')
    print(f'  Recall:       {baseline_metrics["recall_weighted"]:.4f}')
    
    # ========== STEP 2.5: ANALYZE DATASET NOISE ==========
    print('\n' + '='*70)
    print('STEP 2.5: DATASET-LEVEL NOISE ANALYSIS')
    print('='*70)
    
    dataset_analysis = analyze_dataset_noise(X_train_audio, sample_size=8000)
    
    # ========== STEP 2.7: TRAIN ROUTING MODEL (if model-based) ==========
    if use_model_based_routing and train_routing_model_flag:
        print('\n' + '='*70)
        print('STEP 2.7: TRAIN PREPROCESSING ROUTING MODEL')
        print('='*70)
        
        # Train routing model using audio arrays in memory (faster than loading from disk)
        print(f"Using {len(X_train_audio)} training samples to train routing model...")
        
        # Train routing model with pre-trained baseline classifier
        routing_model = train_routing_model(
            X_train_audio,
            y_train,
            n_samples=min(500, len(X_train_audio)),
            classifier_model=baseline_model,
            epochs=15
        )
        
        if routing_model is not None:
            print('\n✓ Routing model trained successfully')
        else:
            print('\n⚠️ Routing model training failed. Will use rule-based fallback.')
    
    # ========== STEP 3: TRAIN ADAPTIVE MODEL ==========
    print('\n' + '='*70)
    print('STEP 3: TRAIN ADAPTIVE MODEL')
    print('='*70)
    
    # ========== METHOD SELECTION ==========
    if use_model_based_routing:
        print(f"\n✓ Using MODEL-BASED method (Meta-Learning Routing)")
        print("   Uses neural network trained on preprocessing performance data")
        print("   Predicts best preprocessing method for each audio sample")
        print(f"   Available methods: {len(METHOD_ID_TO_NAME)} (including {len([k for k in METHOD_ID_TO_NAME if k >= 5])} advanced techniques)")
        selected_method = 'Model-Based (Meta-Learning)'
        pipeline_func = adaptive_pipeline_model_based
    else:
        print(f"\n✓ Using GRANULAR method (Rule-Based Routing)")
        print("   Analyzes each sample with 3 metrics (SNR, Spectral Flatness, Harmonic Ratio)")
        print("   Routes to 6 different preprocessing strategies based on audio characteristics")
        selected_method = 'Granular (Rule-Based)'
        pipeline_func = adaptive_pipeline_granular
    
    print('\nGenerating adaptive features...')
    X_train_adaptive, preprocessing_stats = pipeline_func(X_train_audio)
    X_test_adaptive, _ = pipeline_func(X_test_audio)
    
    print('\nTraining adaptive CNN...')
    adaptive_model = build_genre_cnn(input_shape, num_classes)
    adaptive_model.fit(
        X_train_adaptive, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        verbose=1
    )
    
    print('\nEvaluating adaptive model...')
    y_pred_proba_adaptive = adaptive_model.predict(X_test_adaptive, verbose=0)
    preds_adaptive = np.argmax(y_pred_proba_adaptive, axis=1)
    
    adaptive_metrics = calculate_comprehensive_metrics(y_test, preds_adaptive, y_pred_proba_adaptive, genres)
    
    print(f'\nAdaptive Results:')
    print(f'  Accuracy:     {adaptive_metrics["accuracy"]:.4f} ({adaptive_metrics["accuracy"]*100:.2f}%)')
    print(f'  F1-Score:     {adaptive_metrics["f1_macro"]:.4f}')
    print(f'  AUC-ROC:      {adaptive_metrics["auc_roc_macro"]:.4f}')
    print(f'  Precision:    {adaptive_metrics["precision_weighted"]:.4f}')
    print(f'  Recall:       {adaptive_metrics["recall_weighted"]:.4f}')
    
    # Show preprocessing methods used
    print('\n' + '-'*70)
    print('PREPROCESSING METHODS APPLIED:')
    print('-'*70)
    total_samples = sum(preprocessing_stats.values())
    for method, count in preprocessing_stats.items():
        percentage = (count / total_samples) * 100
        print(f'  {method:20s}: {count:4d} samples ({percentage:5.1f}%)')
    print(f'  Total:              {total_samples:4d} samples')
    
    # ========== STEP 4: COMPARE BASELINE VS ADAPTIVE ==========
    print('\n' + '='*70)
    print('STEP 4: FINAL COMPARISON')
    print('='*70)
    
    # Results table
    print(f'\n{"Metric":<25} {"Baseline":>12} {"Adaptive":>12} {"Improvement":>12}')
    print('-' * 65)
    
    metric_names = [
        ('Accuracy', 'accuracy'),
        ('F1-Score (macro)', 'f1_macro'),
        ('AUC-ROC (macro)', 'auc_roc_macro'),
        ('Precision (weighted)', 'precision_weighted'),
        ('Recall (weighted)', 'recall_weighted')
    ]
    
    for metric_name, metric_key in metric_names:
        baseline_val = baseline_metrics[metric_key]
        adaptive_val = adaptive_metrics[metric_key]
        improvement = (adaptive_val - baseline_val) * 100
        
        print(f'{metric_name:<25} {baseline_val:>12.4f} {adaptive_val:>12.4f} {improvement:>11.2f}%')
    
    # Calculate improvements for file output and validation
    improvement_f1 = (adaptive_metrics['f1_macro'] - baseline_metrics['f1_macro']) * 100
    improvement_acc = (adaptive_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    improvement_auc = (adaptive_metrics['auc_roc_macro'] - baseline_metrics['auc_roc_macro']) * 100
    
    
    # Per-class comparison
    print('\n' + '='*70)
    print('PER-CLASS F1-SCORE COMPARISON')
    print('='*70)
    print(f'\n{"Genre":<15} {"Baseline":>12} {"Adaptive":>12} {"Improvement":>12}')
    print('-' * 55)
    
    for genre in genres:
        baseline_f1 = baseline_metrics['per_class'][genre]['f1']
        adaptive_f1 = adaptive_metrics['per_class'][genre]['f1']
        improvement = (adaptive_f1 - baseline_f1) * 100
        
        print(f'{genre:<15} {baseline_f1:>12.4f} {adaptive_f1:>12.4f} {improvement:>11.2f}%')
    
    # Save results to file
    results_file = os.path.join(RESULTS_DIR, 'baseline_vs_adaptive_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('='*70 + '\n')
        f.write('BASELINE VS ADAPTIVE PREPROCESSING RESULTS\n')
        f.write('='*70 + '\n\n')
        
        f.write('BASELINE (No Preprocessing):\n')
        f.write(f'  Accuracy:     {baseline_metrics["accuracy"]:.4f}\n')
        f.write(f'  F1-Score:     {baseline_metrics["f1_macro"]:.4f}\n')
        f.write(f'  AUC-ROC:      {baseline_metrics["auc_roc_macro"]:.4f}\n')
        f.write(f'  Precision:    {baseline_metrics["precision_weighted"]:.4f}\n')
        f.write(f'  Recall:       {baseline_metrics["recall_weighted"]:.4f}\n\n')
        
        f.write('ADAPTIVE (Smart Routing):\n')
        f.write(f'  Accuracy:     {adaptive_metrics["accuracy"]:.4f}\n')
        f.write(f'  F1-Score:     {adaptive_metrics["f1_macro"]:.4f}\n')
        f.write(f'  AUC-ROC:      {adaptive_metrics["auc_roc_macro"]:.4f}\n')
        f.write(f'  Precision:    {adaptive_metrics["precision_weighted"]:.4f}\n')
        f.write(f'  Recall:       {adaptive_metrics["recall_weighted"]:.4f}\n\n')
        
        f.write('IMPROVEMENTS:\n')
        f.write(f'  F1-Score:     {improvement_f1:+.2f}%\n')
        f.write(f'  Accuracy:     {improvement_acc:+.2f}%\n')
        f.write(f'  AUC-ROC:      {improvement_auc:+.2f}%\n\n')
        
        f.write('DATASET ANALYSIS:\n')
        f.write(f'  Total samples analyzed: {dataset_analysis["total_analyzed"]}\n')
        f.write(f'  Automatically selected method: {selected_method}\n\n')
        
        f.write('  [GRANULAR METHOD] Predicted Preprocessing Routes:\n')
        granular_routing = dataset_analysis['granular_metrics']['routing']
        for method, count in granular_routing.items():
            percentage = (count / dataset_analysis["total_analyzed"]) * 100
            f.write(f'    {method:25s}: {count:4d} ({percentage:5.1f}%)\n')
        
        f.write('\n  [INTELLIGENT METHOD] Predicted Preprocessing Routes:\n')
        intelligent_routing = dataset_analysis['intelligent_metrics']['routing']
        for method, count in intelligent_routing.items():
            percentage = (count / dataset_analysis["total_analyzed"]) * 100
            f.write(f'    {method:25s}: {count:4d} ({percentage:5.1f}%)\n')
        
        f.write('\n  Dataset Quality Metrics:\n')
        f.write(f'    Average SNR:           {dataset_analysis["granular_metrics"]["snr_mean"]:.2f} dB\n')
        f.write(f'    Average Quality Score: {dataset_analysis["intelligent_metrics"]["quality_mean"]:.1f}/100\n')
        f.write(f'    Spectral Flatness:     {dataset_analysis["granular_metrics"]["sf_mean"]:.3f}\n')
        f.write(f'    Harmonic Ratio:        {dataset_analysis["granular_metrics"]["hr_mean"]:.3f}\n')
        
        f.write('\nPREPROCESSING METHODS APPLIED:\n')
        total_samples = sum(preprocessing_stats.values())
        for method, count in preprocessing_stats.items():
            percentage = (count / total_samples) * 100
            f.write(f'  {method:20s}: {count:4d} samples ({percentage:5.1f}%)\n')
        f.write(f'  Total:              {total_samples:4d} samples\n\n')
        
        f.write('PER-CLASS F1-SCORES:\n')
        f.write(f'{"Genre":<15} {"Baseline":>12} {"Adaptive":>12} {"Improvement":>12}\n')
        f.write('-' * 55 + '\n')
        for genre in genres:
            baseline_f1 = baseline_metrics['per_class'][genre]['f1']
            adaptive_f1 = adaptive_metrics['per_class'][genre]['f1']
            improvement = (adaptive_f1 - baseline_f1) * 100
            f.write(f'{genre:<15} {baseline_f1:>12.4f} {adaptive_f1:>12.4f} {improvement:>11.2f}%\n')
    
    print(f'\n✓ Results saved to: {results_file}')
    
    return {'baseline': baseline_metrics, 'adaptive': adaptive_metrics}


if __name__ == '__main__':
    # Run with MODEL-BASED ROUTING (new implementation with meta-learning)
    # Set use_model_based_routing=False to use rule-based granular routing instead
    run_comparative_experiment(
        use_model_based_routing=True,   # Enable neural network routing
        train_routing_model_flag=True   # Train the routing model first
    )


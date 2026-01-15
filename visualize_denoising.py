"""
Visualization Tool: Before/After Denoising Comparison

Creates side-by-side spectrograms showing the effect of different denoising methods.
Useful for demo presentations and analysis.
"""
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob

# Import denoisers from POC
import sys
sys.path.append('.')

SR = 22050
N_FFT = 2048
HOP_LENGTH = 512

# Denoising functions (copied from POC for standalone use)
def denoise_spectral_subtraction(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Spectral subtraction for stationary noise."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_mag = np.percentile(mag, 5, axis=1, keepdims=True)
    alpha = 0.5
    mag_clean = np.maximum(mag - alpha * noise_mag, 0.5 * mag)
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    return audio_clean


def denoise_median_filtering(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Median filtering for non-stationary noise."""
    from scipy.signal import medfilt2d
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    mag_med = medfilt2d(mag, kernel_size=(3, 5))
    alpha = 0.6
    mag_clean = alpha * mag_med + (1 - alpha) * mag
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    return audio_clean


def denoise_wiener_filter(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Wiener filtering."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    energy = np.sum(mag, axis=0)
    noise_frames = energy < np.percentile(energy, 5)
    if np.sum(noise_frames) > 0:
        noise_mag = np.median(mag[:, noise_frames], axis=1, keepdims=True)
    else:
        noise_mag = np.percentile(mag, 5, axis=1, keepdims=True)
    signal_power = mag ** 2
    noise_power = noise_mag ** 2
    wiener_gain = signal_power / (signal_power + noise_power + 1e-8)
    wiener_gain = np.clip(wiener_gain, 0.5, 1.0)
    mag_clean = mag * wiener_gain
    stft_clean = mag_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean, hop_length=hop_length, length=len(audio))
    return audio_clean


def characterize_noise_type(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Determine noise type from audio characteristics."""
    sf = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    mean_sf = float(np.mean(sf))
    sc = librosa.feature.spectral_centroid(y=audio, sr=SR, n_fft=n_fft, hop_length=hop_length)
    mean_sc = float(np.mean(sc))
    zcr = librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)
    mean_zcr = float(np.mean(zcr))
    
    if mean_sf > 0.75:
        return 'broadband', mean_sf, mean_sc, mean_zcr
    elif mean_sc < 1500:
        return 'lowfreq', mean_sf, mean_sc, mean_zcr
    elif mean_zcr > 0.20:
        return 'transient', mean_sf, mean_sc, mean_zcr
    else:
        return 'general', mean_sf, mean_sc, mean_zcr


def plot_comparison(audio_path, output_path=None, duration=10):
    """Create comprehensive before/after denoising comparison."""
    # Load audio
    print(f'Processing: {os.path.basename(audio_path)}')
    audio, sr = librosa.load(audio_path, sr=SR, duration=duration)
    
    # Characterize noise
    noise_type, sf, sc, zcr = characterize_noise_type(audio)
    
    # Apply appropriate denoising
    if noise_type == 'broadband':
        audio_denoised = denoise_spectral_subtraction(audio)
        method_name = 'Spectral Subtraction'
    elif noise_type in ['lowfreq', 'transient']:
        audio_denoised = denoise_median_filtering(audio)
        method_name = 'Median Filtering'
    else:
        audio_denoised = denoise_wiener_filter(audio)
        method_name = 'Wiener Filtering'
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)
    
    # Original waveform
    ax1 = fig.add_subplot(gs[0, 0])
    librosa.display.waveshow(audio, sr=sr, ax=ax1, color='steelblue')
    ax1.set_title('Original Waveform', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # Denoised waveform
    ax2 = fig.add_subplot(gs[0, 1])
    librosa.display.waveshow(audio_denoised, sr=sr, ax=ax2, color='forestgreen')
    ax2.set_title(f'Denoised Waveform ({method_name})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # Original spectrogram
    ax3 = fig.add_subplot(gs[1, 0])
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img1 = librosa.display.specshow(D_orig, sr=sr, hop_length=HOP_LENGTH, 
                                     x_axis='time', y_axis='hz', ax=ax3, cmap='viridis')
    ax3.set_title('Original Spectrogram', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency (Hz)')
    plt.colorbar(img1, ax=ax3, format='%+2.0f dB')
    
    # Denoised spectrogram
    ax4 = fig.add_subplot(gs[1, 1])
    D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(audio_denoised)), ref=np.max)
    img2 = librosa.display.specshow(D_clean, sr=sr, hop_length=HOP_LENGTH,
                                     x_axis='time', y_axis='hz', ax=ax4, cmap='viridis')
    ax4.set_title('Denoised Spectrogram', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency (Hz)')
    plt.colorbar(img2, ax=ax4, format='%+2.0f dB')
    
    # Difference spectrogram (what was removed)
    ax5 = fig.add_subplot(gs[2, :])
    D_diff = D_orig - D_clean
    img3 = librosa.display.specshow(D_diff, sr=sr, hop_length=HOP_LENGTH,
                                     x_axis='time', y_axis='hz', ax=ax5, cmap='coolwarm',
                                     vmin=-20, vmax=20)
    ax5.set_title('Difference (Removed Components)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency (Hz)')
    ax5.set_xlabel('Time (s)')
    plt.colorbar(img3, ax=ax5, format='%+2.0f dB', label='dB difference')
    
    # Add metadata text
    metadata_text = (
        f'File: {os.path.basename(audio_path)}\n'
        f'Detected Noise Type: {noise_type.upper()}\n'
        f'Method Applied: {method_name}\n'
        f'Spectral Flatness: {sf:.3f}\n'
        f'Spectral Centroid: {sc:.1f} Hz\n'
        f'Zero-Crossing Rate: {zcr:.3f}'
    )
    fig.text(0.02, 0.98, metadata_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             family='monospace')
    
    fig.suptitle('Adaptive Denoising Analysis', fontsize=16, fontweight='bold', y=0.995)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')
    else:
        plt.show()
    
    plt.close()


def visualize_dataset_samples(dataset_dir, num_samples=5, output_dir='./visualizations'):
    """Create visualizations for multiple random samples from dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find audio files
    audio_files = []
    for ext in ['*.mp3', '*.wav']:
        audio_files.extend(glob.glob(os.path.join(dataset_dir, '**', ext), recursive=True))
    
    if not audio_files:
        print(f'No audio files found in {dataset_dir}')
        return
    
    print(f'Found {len(audio_files)} audio files')
    
    # Sample randomly
    import random
    random.seed(42)
    samples = random.sample(audio_files, min(num_samples, len(audio_files)))
    
    print(f'Creating visualizations for {len(samples)} samples...')
    for i, audio_path in enumerate(samples, 1):
        output_path = os.path.join(output_dir, f'denoising_comparison_{i}.png')
        try:
            plot_comparison(audio_path, output_path, duration=10)
        except Exception as e:
            print(f'Error processing {audio_path}: {e}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize denoising effects')
    parser.add_argument('--file', type=str, help='Single audio file to visualize')
    parser.add_argument('--dataset', type=str, help='Dataset directory to sample from')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples from dataset')
    parser.add_argument('--output-dir', type=str, default='./visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    if args.file:
        # Single file mode
        output_path = os.path.join(args.output_dir, 'denoising_comparison.png')
        os.makedirs(args.output_dir, exist_ok=True)
        plot_comparison(args.file, output_path)
    
    elif args.dataset:
        # Dataset sampling mode
        visualize_dataset_samples(args.dataset, args.num_samples, args.output_dir)
    
    else:
        # Default: sample from FMA-medium
        fma_dir = os.path.join('.', 'datasets', 'fma_medium')
        if os.path.exists(fma_dir):
            print('Using FMA-medium dataset...')
            visualize_dataset_samples(fma_dir, num_samples=5, output_dir=args.output_dir)
        else:
            print('Please specify --file or --dataset')
            parser.print_help()

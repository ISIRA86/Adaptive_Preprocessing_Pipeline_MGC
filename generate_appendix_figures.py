"""
generate_appendix_figures.py
----------------------------
Generates all figures and tables needed for Appendix F testing chapter:

  1. SNR improvement table  — avg SNR before vs. after each preprocessing method
  2. Mel-spectrogram comparison figures — one figure per method (before / after)
  3. Spectral flatness / harmonic ratio / SNR distribution plots
  4. Routing model figures:
       a) routing_distribution.png      — bar chart: samples assigned to each method
       b) routing_confusion_matrix.png  — oracle labels vs model predictions
       c) routing_feature_importance.png — permutation importance for all 10 features
       d) routing_training_curves.png   — accuracy & loss vs epoch

Run from the project root (with the venv active):
    python generate_appendix_figures.py

Output is written to:
    results/appendix/snr_table.csv
    results/appendix/snr_improvement.png
    results/appendix/spectrograms/<method>_comparison.png
    results/appendix/spectral_flatness_distribution.png
    results/appendix/harmonic_ratio_distribution.png
    results/appendix/snr_distribution.png
    results/appendix/routing_distribution.png
    results/appendix/routing_confusion_matrix.png
    results/appendix/routing_feature_importance.png
    results/appendix/routing_training_curves.png
"""

import os
import sys
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ── Import from the framework ─────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adaptive_preprocessing_framework import (
    load_fma_dataset,
    apply_preprocessing_method,
    generate_routing_training_data,
    train_routing_model,
    adaptive_pipeline_model_based,
    METHOD_ID_TO_NAME,
    SR, N_MELS, N_FFT, HOP_LENGTH,
    RESULTS_DIR,
)
import adaptive_preprocessing_framework as _fw   # to read _ROUTING_FEAT_MEAN/_STD after training

# ── Output directory ──────────────────────────────────────────────────
OUT_DIR = os.path.join(RESULTS_DIR, 'appendix')
SPEC_DIR = os.path.join(OUT_DIR, 'spectrograms')
os.makedirs(SPEC_DIR, exist_ok=True)

# ── How many audio samples to use for the SNR / distribution analysis ─
ANALYSIS_SAMPLES = 200   # increase for a more representative table; 200 is fast


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

def compute_snr(audio: np.ndarray) -> float:
    """Simple broadband SNR estimate (signal power / noise-floor power)."""
    signal_power = np.mean(audio ** 2)
    noise_estimate = np.std(audio[:min(1000, len(audio) // 4)])
    if signal_power < 1e-10 or noise_estimate < 1e-10:
        return 30.0          # treat silence as high SNR
    snr = 10 * np.log10(signal_power / (noise_estimate ** 2 + 1e-10))
    return float(np.clip(snr, -20, 60))


def compute_spectral_flatness(audio: np.ndarray) -> float:
    sf = librosa.feature.spectral_flatness(y=audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return float(np.mean(sf))


def compute_harmonic_ratio(audio: np.ndarray) -> float:
    try:
        harmonic, _ = librosa.effects.hpss(audio, margin=3.0)
        return float(np.sum(harmonic ** 2) / (np.sum(audio ** 2) + 1e-8))
    except Exception:
        return 0.5


def mel_db(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max)


# ─────────────────────────────────────────────────────────────────────
# 1.  SNR before / after table
# ─────────────────────────────────────────────────────────────────────

def generate_snr_table(audio_batch: list) -> pd.DataFrame:
    print(f'\n[1/3] Computing SNR before/after for {len(audio_batch)} samples ...')

    rows = []
    for method_id, method_name in METHOD_ID_TO_NAME.items():
        snr_before_list, snr_after_list = [], []

        for audio in audio_batch:
            snr_before_list.append(compute_snr(audio))
            processed = apply_preprocessing_method(audio, method_id)
            snr_after_list.append(compute_snr(processed))

        snr_before = np.mean(snr_before_list)
        snr_after  = np.mean(snr_after_list)
        improvement = snr_after - snr_before

        rows.append({
            'Method ID': method_id,
            'Method Name': method_name,
            'Avg SNR Before (dB)': round(snr_before, 2),
            'Avg SNR After (dB)':  round(snr_after,  2),
            'SNR Improvement (dB)': round(improvement, 2),
        })

        print(f'  [{method_id}] {method_name:<22}  before={snr_before:+.2f} dB  '
              f'after={snr_after:+.2f} dB  Δ={improvement:+.2f} dB')

    df = pd.DataFrame(rows)

    # ── save CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, 'snr_table.csv')
    df.to_csv(csv_path, index=False)
    print(f'  Saved SNR table → {csv_path}')

    # ── save bar chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(rows))
    width = 0.35
    bars_b = ax.bar(x - width / 2, df['Avg SNR Before (dB)'], width,
                    label='Before preprocessing', color='#5b9bd5', alpha=0.85)
    bars_a = ax.bar(x + width / 2, df['Avg SNR After (dB)'],  width,
                    label='After preprocessing',  color='#ed7d31', alpha=0.85)

    # SNR improvement labels above the "after" bars
    for bar, imp in zip(bars_a, df['SNR Improvement (dB)']):
        ha = bar.get_x() + bar.get_width() / 2
        ax.text(ha, bar.get_height() + 0.4, f'Δ{imp:+.2f}',
                ha='center', va='bottom', fontsize=8, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(df['Method Name'], rotation=20, ha='right')
    ax.set_ylabel('Average SNR (dB)')
    ax.set_title('Average SNR Before vs. After Preprocessing — per Method')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, 'snr_improvement.png')
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f'  Saved SNR bar chart → {png_path}')

    return df


# ─────────────────────────────────────────────────────────────────────
# 2.  Mel-spectrogram comparison figures
# ─────────────────────────────────────────────────────────────────────

def generate_spectrogram_comparisons(audio_sample: np.ndarray, sample_idx: int = 0):
    """
    Generate one before/after mel-spectrogram figure per preprocessing method,
    plus one figure that shows all methods side-by-side.
    """
    print(f'\n[2/3] Generating mel-spectrogram comparisons (sample idx={sample_idx}) ...')

    original_db = mel_db(audio_sample)

    # ── per-method figure ─────────────────────────────────────────────
    for method_id, method_name in METHOD_ID_TO_NAME.items():
        processed = apply_preprocessing_method(audio_sample, method_id)
        processed_db = mel_db(processed)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

        im0 = axes[0].imshow(original_db, aspect='auto', origin='lower',
                             cmap='magma', vmin=-80, vmax=0)
        axes[0].set_title('Original (no preprocessing)', fontsize=11)
        axes[0].set_xlabel('Time frames')
        axes[0].set_ylabel('Mel frequency bins')
        plt.colorbar(im0, ax=axes[0], format='%+2.0f dB', fraction=0.046)

        im1 = axes[1].imshow(processed_db, aspect='auto', origin='lower',
                             cmap='magma', vmin=-80, vmax=0)
        label = method_name if method_name != 'none' else 'none (baseline)'
        axes[1].set_title(f'After: {label}', fontsize=11)
        axes[1].set_xlabel('Time frames')
        plt.colorbar(im1, ax=axes[1], format='%+2.0f dB', fraction=0.046)

        fig.suptitle(f'Mel-Spectrogram Comparison — Method: {method_name}', fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(SPEC_DIR, f'{method_name}_comparison.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'  Saved → {out_path}')

    # ── combined overview figure (all methods in one grid) ─────────────
    n_methods = len(METHOD_ID_TO_NAME)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4), sharey=True)

    # first column = original
    axes[0].imshow(original_db, aspect='auto', origin='lower',
                   cmap='magma', vmin=-80, vmax=0)
    axes[0].set_title('Original', fontsize=9)
    axes[0].set_ylabel('Mel bins')
    axes[0].set_xlabel('Time')

    for col, (method_id, method_name) in enumerate(METHOD_ID_TO_NAME.items(), start=1):
        processed_db = mel_db(apply_preprocessing_method(audio_sample, method_id))
        axes[col].imshow(processed_db, aspect='auto', origin='lower',
                         cmap='magma', vmin=-80, vmax=0)
        axes[col].set_title(method_name, fontsize=9)
        axes[col].set_xlabel('Time')

    fig.suptitle('Mel-Spectrogram: Original vs. All Preprocessing Methods', fontsize=11)
    plt.tight_layout()
    overview_path = os.path.join(SPEC_DIR, 'all_methods_overview.png')
    fig.savefig(overview_path, dpi=150)
    plt.close(fig)
    print(f'  Saved overview → {overview_path}')


# ─────────────────────────────────────────────────────────────────────
# 3.  Spectral flatness & harmonic ratio distributions
# ─────────────────────────────────────────────────────────────────────

def generate_distribution_plots(audio_batch: list):
    print(f'\n[3/3] Computing audio characteristic distributions ({len(audio_batch)} samples) ...')

    sf_values, hr_values, snr_values = [], [], []

    for i, audio in enumerate(audio_batch):
        sf_values.append(compute_spectral_flatness(audio))
        hr_values.append(compute_harmonic_ratio(audio))
        snr_values.append(compute_snr(audio))
        if (i + 1) % 50 == 0:
            print(f'  Processed {i + 1}/{len(audio_batch)} samples ...')

    sf  = np.array(sf_values)
    hr  = np.array(hr_values)
    snr = np.array(snr_values)

    # ── spectral flatness ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sf, bins=40, color='#5b9bd5', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(sf),   color='red',    linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(sf):.3f}')
    ax.axvline(np.median(sf), color='orange', linestyle=':',  linewidth=1.5,
               label=f'Median = {np.median(sf):.3f}')
    ax.set_xlabel('Spectral Flatness  (0 = tonal / harmonic,  1 = noise-like)')
    ax.set_ylabel('Number of samples')
    ax.set_title('Spectral Flatness Distribution — FMA Dataset')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    sf_path = os.path.join(OUT_DIR, 'spectral_flatness_distribution.png')
    fig.savefig(sf_path, dpi=150)
    plt.close(fig)
    print(f'  Saved → {sf_path}')

    # ── harmonic ratio ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(hr, bins=40, color='#ed7d31', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(hr),   color='blue',  linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(hr):.3f}')
    ax.axvline(np.median(hr), color='green', linestyle=':',  linewidth=1.5,
               label=f'Median = {np.median(hr):.3f}')
    ax.set_xlabel('Harmonic Ratio  (0 = purely percussive / noisy,  1 = purely harmonic)')
    ax.set_ylabel('Number of samples')
    ax.set_title('Harmonic Ratio Distribution — FMA Dataset')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    hr_path = os.path.join(OUT_DIR, 'harmonic_ratio_distribution.png')
    fig.savefig(hr_path, dpi=150)
    plt.close(fig)
    print(f'  Saved → {hr_path}')

    # ── SNR distribution ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(snr, bins=40, color='#70ad47', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(snr),   color='red',    linestyle='--', linewidth=1.5,
               label=f'Mean = {np.mean(snr):.1f} dB')
    ax.axvline(np.median(snr), color='orange', linestyle=':',  linewidth=1.5,
               label=f'Median = {np.median(snr):.1f} dB')
    ax.set_xlabel('Estimated SNR (dB)')
    ax.set_ylabel('Number of samples')
    ax.set_title('Signal-to-Noise Ratio Distribution — FMA Dataset (before preprocessing)')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    snr_path = os.path.join(OUT_DIR, 'snr_distribution.png')
    fig.savefig(snr_path, dpi=150)
    plt.close(fig)
    print(f'  Saved → {snr_path}')

    # ── print summary stats ───────────────────────────────────────────
    print('\n  Summary statistics:')
    for name, arr in [('Spectral Flatness', sf), ('Harmonic Ratio', hr), ('SNR (dB)', snr)]:
        print(f'  {name:<22} mean={np.mean(arr):.3f}  median={np.median(arr):.3f}'
              f'  min={np.min(arr):.3f}  max={np.max(arr):.3f}')


# ─────────────────────────────────────────────────────────────────────
# 4.  Routing model figures
#     a) Routing distribution bar chart
#     b) Confusion matrix  (oracle labels vs model predictions)
#     c) Feature importance (permutation-based)
#     d) Training curves   (accuracy & loss vs epoch)
# ─────────────────────────────────────────────────────────────────────

# The 10 feature names produced by extract_audio_features_for_routing
ROUTING_FEATURE_NAMES = [
    'SNR (dB)',
    'Spectral Flatness (mean)',
    'Spectral Flatness (std)',
    'Spectral Centroid (norm)',
    'Zero Crossing Rate',
    'Harmonic Ratio',
    'Percussive Ratio',
    'RMS Energy (mean)',
    'RMS Energy (std)',
    'Spectral Bandwidth (norm)',
]

# How many samples to use for routing model training (keep low for speed)
ROUTING_SAMPLES = 150


def _routing_distribution_chart(method_stats: dict):
    """Bar chart of how many samples were routed to each preprocessing method."""
    methods = list(METHOD_ID_TO_NAME.values())
    counts  = [method_stats.get(m, 0) for m in methods]
    total   = max(sum(counts), 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(methods, counts, color='#5b9bd5', edgecolor='white', alpha=0.85)

    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Preprocessing Method')
    ax.set_ylabel('Number of samples assigned')
    ax.set_title('Routing Distribution — Samples Assigned to Each Preprocessing Method')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'routing_distribution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved → {path}')


def _routing_confusion_matrix(X_raw: np.ndarray, y_oracle_ids: np.ndarray):
    """Confusion matrix: oracle labels vs routing model predictions."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Normalise with the stats saved during training
    feat_mean = _fw._ROUTING_FEAT_MEAN
    feat_std  = _fw._ROUTING_FEAT_STD
    X_norm = (X_raw - feat_mean) / (feat_std + 1e-8)

    preds_proba = _fw.ROUTING_MODEL.predict(X_norm, verbose=0)
    y_pred_ids  = np.argmax(preds_proba, axis=1)

    methods_present = sorted(set(y_oracle_ids.tolist()) | set(y_pred_ids.tolist()))
    labels_present  = [METHOD_ID_TO_NAME[m] for m in methods_present]

    cm = confusion_matrix(y_oracle_ids, y_pred_ids, labels=methods_present)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, label='Proportion')

    tick_marks = np.arange(len(labels_present))
    ax.set_xticks(tick_marks); ax.set_xticklabels(labels_present, rotation=30, ha='right')
    ax.set_yticks(tick_marks); ax.set_yticklabels(labels_present)

    thresh = 0.5
    for i in range(len(labels_present)):
        for j in range(len(labels_present)):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black', fontsize=9)

    ax.set_xlabel('Predicted method')
    ax.set_ylabel('Oracle (true best) method')
    ax.set_title('Routing Model Confusion Matrix\n(row-normalised)')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'routing_confusion_matrix.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved → {path}')


def _routing_feature_importance(X_raw: np.ndarray, y_oracle_ids: np.ndarray,
                                n_repeats: int = 8):
    """Permutation-based feature importance for the routing neural network."""
    feat_mean = _fw._ROUTING_FEAT_MEAN
    feat_std  = _fw._ROUTING_FEAT_STD
    X_norm = (X_raw - feat_mean) / (feat_std + 1e-8)

    # Baseline accuracy
    preds_base = np.argmax(_fw.ROUTING_MODEL.predict(X_norm, verbose=0), axis=1)
    base_acc   = float(np.mean(preds_base == y_oracle_ids))

    importances = []
    for feat_idx in range(X_norm.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_norm.copy()
            np.random.shuffle(X_perm[:, feat_idx])          # permute this feature
            preds_perm = np.argmax(_fw.ROUTING_MODEL.predict(X_perm, verbose=0), axis=1)
            drops.append(base_acc - float(np.mean(preds_perm == y_oracle_ids)))
        importances.append(float(np.mean(drops)))

    # Sort descending
    order = np.argsort(importances)[::-1]
    sorted_names = [ROUTING_FEATURE_NAMES[i] for i in order]
    sorted_imp   = [importances[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#c00000' if v > 0 else '#a0a0a0' for v in sorted_imp]
    ax.barh(sorted_names[::-1], sorted_imp[::-1], color=colors[::-1],
            edgecolor='white', alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Mean accuracy drop when feature is permuted\n(higher = more important)')
    ax.set_title('Routing Model — Permutation Feature Importance')
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'routing_feature_importance.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved → {path}')

    # Print table
    print('  Feature importance (accuracy drop):')
    for name, imp in zip(sorted_names, sorted_imp):
        print(f'    {name:<32} {imp:+.4f}')


def _routing_training_curves(history):
    """Accuracy and loss vs epoch for the routing neural network."""
    epochs = range(1, len(history.history['accuracy']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(epochs, history.history['accuracy'],     'o-', color='#5b9bd5',
                 label='Train accuracy',      linewidth=1.8, markersize=4)
    axes[0].plot(epochs, history.history['val_accuracy'], 's--', color='#ed7d31',
                 label='Validation accuracy', linewidth=1.8, markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Routing Model — Accuracy vs Epoch')
    axes[0].legend()
    axes[0].grid(linestyle='--', alpha=0.4)

    # Loss
    axes[1].plot(epochs, history.history['loss'],     'o-', color='#5b9bd5',
                 label='Train loss',      linewidth=1.8, markersize=4)
    axes[1].plot(epochs, history.history['val_loss'], 's--', color='#ed7d31',
                 label='Validation loss', linewidth=1.8, markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Categorical cross-entropy loss')
    axes[1].set_title('Routing Model — Loss vs Epoch')
    axes[1].legend()
    axes[1].grid(linestyle='--', alpha=0.4)

    plt.suptitle('Routing Neural Network Training History', fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'routing_training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {path}')


def generate_routing_figures(audio_batch: list, labels: np.ndarray):
    print(f'\n[4/4] Generating routing model figures ({ROUTING_SAMPLES} samples) ...')

    # ── a) Train the routing model (no classifier oracle — uses SNR proxy) ──
    print('  Training routing model ...')
    _, _, history = train_routing_model(
        audio_batch, labels,
        n_samples=ROUTING_SAMPLES,
        classifier_model=None,   # SNR-based oracle; fast, no CNN needed
        epochs=20,
    )

    # ── Routing distribution (apply to the full batch) ─────────────────
    print('  Running adaptive pipeline for routing distribution ...')
    _, method_stats = adaptive_pipeline_model_based(audio_batch[:ROUTING_SAMPLES])
    _routing_distribution_chart(method_stats)

    # ── Confusion matrix & feature importance ─────────────────────────
    # Re-generate oracle labels on the same subset so we have ground truth
    print('  Re-generating oracle labels for confusion matrix ...')
    X_raw, y_cat = generate_routing_training_data(
        audio_batch, labels,
        n_samples=ROUTING_SAMPLES,
        classifier_model=None,
    )
    y_oracle_ids = np.argmax(y_cat, axis=1)

    _routing_confusion_matrix(X_raw, y_oracle_ids)
    _routing_feature_importance(X_raw, y_oracle_ids)

    # ── Training curves ───────────────────────────────────────────────
    _routing_training_curves(history)


# ─────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('Appendix Figure Generator')
    print('=' * 60)

    # Load a small slice of the dataset
    print(f'\nLoading dataset (up to {ANALYSIS_SAMPLES} samples) ...')
    audio_batch, labels, _ = load_fma_dataset(max_samples=ANALYSIS_SAMPLES)
    print(f'Loaded {len(audio_batch)} audio samples.')

    if len(audio_batch) == 0:
        print('ERROR: No audio samples loaded. Check your dataset path.')
        return

    # 1 – SNR table
    snr_df = generate_snr_table(audio_batch)

    # 2 – Spectrogram comparisons (use the first sample)
    generate_spectrogram_comparisons(audio_batch[0], sample_idx=0)

    # 3 – Distribution plots
    generate_distribution_plots(audio_batch)

    # 4 – Routing model figures
    generate_routing_figures(audio_batch, labels)

    print('\n' + '=' * 60)
    print(f'All outputs saved to: {OUT_DIR}')
    print('=' * 60)

    # Pretty-print the SNR table to terminal
    print('\nSNR Table (for copy-paste into appendix):')
    print(snr_df.to_string(index=False))


if __name__ == '__main__':
    main()

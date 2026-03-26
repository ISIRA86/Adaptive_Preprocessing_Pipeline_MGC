# POC: Adaptive Denoising for Audio Classification

This repository contains **three POC implementations** demonstrating adaptive noise-reduction preprocessing for audio classification tasks.

## Available POCs

### 1. GTZAN POC (`poc_mgc_adaptive.py`)
- **Dataset**: GTZAN (800-1000 clips, 10 genres)
- **Noise**: Synthetic (programmatically added hiss/rumble)
- **Use case**: Controlled experiments
- **Status**: Completed, local dataset configured

### 2. FMA POC (`poc_fma_adaptive.py`) ⭐ **RECOMMENDED**
- **Dataset**: Free Music Archive - FMA small (8,000 tracks, 8 genres)
- **Noise**: Natural recording artifacts and compression noise
- **Use case**: Real-world music classification with realistic noise
- **Advantages**: 
  - No synthetic noise needed
  - Larger dataset than GTZAN
  - 3 denoising strategies (spectral, median, Wiener)
  - Enhanced multi-feature noise characterization
- **Download**: https://github.com/mdeff/fma

### 3. AudioSet POC (`poc_audioset_adaptive.py`)
- **Dataset**: AudioSet balanced (22k+ clips, 10-second YouTube audio)
- **Noise**: Real-world environmental noise
- **Use case**: Most challenging, diverse audio events
- **Advantages**:
  - Most realistic noise scenarios
  - Includes harmonic-percussive separation
  - Adaptive strategy selection
- **Download**: https://research.google.com/audioset/

---

## Comparison Matrix

| POC       | Clips  | Duration | Noise Type     | Realism | Setup  | Training Time |
|-----------|--------|----------|----------------|---------|--------|---------------|
| GTZAN     | 800    | 30s      | Synthetic      | Low     | Easy   | ~1-2 hours    |
| FMA       | 8,000  | 30s      | Natural        | High    | Medium | ~8-12 hours   |
| AudioSet  | 22,000 | 10s      | Environmental  | Highest | Hard   | ~20+ hours    |

---

## Quick Start

### Option 1: FMA POC (Recommended for realistic results)

```powershell
# 1. Download FMA-small dataset (~8GB)
#    URL: https://os.unil.cloud.switch.ch/fma/fma_small.zip
#    Extract to: ./datasets/fma_small/

# 2. Download FMA metadata (~300MB)
#    URL: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
#    Extract tracks.csv to: ./datasets/fma_metadata/

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the POC
python poc_fma_adaptive.py

# Quick test with subset (1000 samples, ~1-2 hours):
# Edit poc_fma_adaptive.py: MAX_SAMPLES = 1000
```

### Option 2: GTZAN POC (Already configured)

```powershell
# Already set up with local Kaggle dataset
python poc_mgc_adaptive.py

# For quick testing (200 samples, ~20 mins):
# Already configured: MAX_SAMPLES = 200
```

### Option 3: AudioSet POC (Advanced)

```powershell
# 1. Download AudioSet (pre-processed recommended)
#    Option A: Kaggle AudioSet dataset
#    Option B: Academic download with youtube-dl
#    Place audio files in: ./datasets/audioset/audio/

# 2. Run
python poc_audioset_adaptive.py
```

---

## Installation

### Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- numpy>=1.22,<2.0
- librosa>=0.11.0
- tensorflow>=2.7.0
- scikit-learn>=1.1.0
- scipy>=1.13.0
- soundfile>=0.12.1
- numba>=0.60.0
- pandas>=1.3.0

---

## Architecture Overview

All POCs follow the same experimental design:

```
┌──────────────┐
│ Audio Dataset│
└──────┬───────┘
       │
       ├─────────────────────┬─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
┌─────────────┐      ┌──────────────┐     ┌──────────────┐
│  Baseline   │      │   Adaptive   │     │  Test Data   │
│  Pipeline   │      │   Pipeline   │     │   Split      │
│ (No Denoise)│      │ (Denoise)    │     └──────────────┘
└──────┬──────┘      └──────┬───────┘
       │                    │
       │  ┌─────────────────┤
       │  │ Noise Detection │
       │  └─────────┬───────┘
       │            │
       │            ├── Spectral Subtraction (broadband)
       │            ├── Median Filtering (transient)
       │            └── Wiener/HPSS (harmonic)
       │                    │
       ▼                    ▼
┌─────────────┐      ┌──────────────┐
│   CNN Model │      │  CNN Model   │
│  (Baseline) │      │  (Adaptive)  │
└──────┬──────┘      └──────┬───────┘
       │                    │
       ▼                    ▼
┌─────────────────────────────────┐
│   Compare Test Accuracies       │
│   Generate Classification       │
│   Reports                        │
└─────────────────────────────────┘
```

---

## 📈 Expected Results

### With Synthetic Noise (GTZAN)
- **Baseline**: 40-60% accuracy
- **Adaptive**: 50-70% accuracy
- **Improvement**: ~10-15% with strong noise (SNR=15dB)

### With Natural Noise (FMA, AudioSet)
- **Baseline**: 50-65% accuracy
- **Adaptive**: 58-72% accuracy
- **Improvement**: ~5-12% depending on noise levels

---

## 🔬 Key Innovations in New POCs

1. **Real-world noise**: No synthetic injection, natural recording artifacts
2. **Multi-strategy denoising**:
   - Spectral subtraction (stationary noise)
   - Median filtering (transient/impulsive)
   - Wiener filtering (adaptive SNR-based)
   - Harmonic-percussive separation (music-specific)

3. **Smart routing**: Multi-feature analysis
   - Spectral flatness (noise type)
   - Spectral centroid (frequency distribution)
   - Spectral bandwidth (spread)
   - Zero-crossing rate (temporal characteristics)
   - RMS energy variance (dynamics)

4. **Diagnostic output**: Real-time strategy selection logging

---

## 📝 Customization

### Adjust Sample Size

```python
# In any POC file
MAX_SAMPLES = 500  # Use 500 samples for quick testing
MAX_SAMPLES = None # Use full dataset
```

### Adjust Training

```python
EPOCHS = 5         # Faster, less accurate
EPOCHS = 15        # Slower, more accurate
BATCH_SIZE = 32    # GPU recommended
```

### Adjust Noise Levels (GTZAN only)

```python
# In poc_mgc_adaptive.py
snr_db=10  # Very noisy (harder)
snr_db=20  # Moderate noise
snr_db=30  # Light noise (easier)
```

---

## 🐛 Troubleshooting

### "Dataset not found"
- Check directory structure matches expected paths
- See individual POC file headers for exact structure

### "Out of memory" 
- Reduce `MAX_SAMPLES`
- Reduce `BATCH_SIZE`
- Use smaller mel-spectrogram dimensions

### Slow training
- Use GPU (install `tensorflow-gpu`)
- Reduce `MAX_SAMPLES` or `EPOCHS`
- Use smaller dataset (GTZAN subset)

### No improvement observed
- Increase noise levels (GTZAN: lower SNR)
- Check if dataset already has noise
- Verify denoising is being applied (check logs)

---

## 📚 References

- **GTZAN**: http://marsyas.info/downloads/datasets.html
- **FMA**: https://github.com/mdeff/fma
- **AudioSet**: https://research.google.com/audioset/

---

## 🎯 Next Steps

After running POCs:
1. Compare results across all three datasets
2. Tune denoising parameters for best performance
3. Export trained models for deployment
4. Extend to real-time audio processing
5. Implement custom noise classifiers (replace heuristics with ML)

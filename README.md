# Adaptive Noise-Aware Preprocessing for Music Genre Classification

A comprehensive framework for intelligent, noise-aware preprocessing to improve music genre classification on real-world datasets using advanced signal processing and machine learning techniques.

## Overview

This project develops and evaluates a noise-aware preprocessing pipeline specifically designed for music genre classification. The framework intelligently analyzes audio characteristics and applies appropriate denoising methods, achieving **5.76% improvement** over baseline on the FMA-medium dataset.

**Key Achievement**: Validated adaptive preprocessing improves classification accuracy from 45.22% (baseline) to 50.98% (adaptive) on full FMA-medium dataset (22,930 training samples).

## Features

### Advanced Preprocessing Methods
- **Spectral Gating**: Smooth sigmoid-based noise gate for broadband noise reduction
- **HPSS**: Harmonic-Percussive Source Separation for music structure extraction
- **Spleeter**: Deep learning-based vocal/accompaniment separation (2-stem)
- **LUFS Normalization**: EBU R128 standard loudness normalization (-23 LUFS target)
- **Demucs Support**: Facebook's state-of-the-art hybrid transformer denoising (optional)
- **Classical Methods**: Spectral subtraction, median filtering, Wiener filtering

### Intelligent Adaptive Routing
Automatically characterizes audio and selects optimal preprocessing methods based on:
- **Spectral Flatness**: Detects broadband noise vs tonal content
- **Spectral Centroid**: Identifies frequency distribution characteristics
- **Zero-Crossing Rate**: Measures temporal complexity
- **Audio Type Classification**: Routes to broadband_noise/harmonic_rich/vocal_heavy/general pipelines

### Comprehensive Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-score (macro/weighted), AUC-ROC
- **Per-Class Analysis**: Detailed breakdown for all 8 genre classes
- **Visualization**: Before/after mel-spectrogram comparison
- **Requirement Validation**: Automatic verification of >4% improvement threshold

### Analysis & Reporting
- Automated performance comparison charts
- Per-genre F1 score analysis
- Confusion matrices
- Processing time benchmarks

## Project Structure

```
.
├── adaptive_preprocessing_framework.py  # Comprehensive framework (Week 4-5)
├── poc_fma_adaptive.py                 # Phase 1 validated baseline
├── poc_audioset_adaptive.py            # AudioSet integration
├── download_audioset_subset.py         # Music-focused AudioSet filtering
├── analyze_results.py                  # Results analysis and visualization
├── requirements.txt                    # Core dependencies
├── requirements_advanced.txt           # Advanced method dependencies
├── Adaptive_MGC_POC.ipynb             # Interactive notebook
├── results/                            # Phase 1 results
├── results_advanced/                   # Advanced framework results
└── README.md                           # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ISIRA86/Adaptive_Preprocessing_Pipeline_MGC.git
cd Adaptive_Preprocessing_Pipeline_MGC
```

2. Install core dependencies:
```bash
pip install -r requirements.txt
```

3. Install advanced preprocessing dependencies:
```bash
pip install -r requirements_advanced.txt
```

4. Download datasets:
   - **FMA-medium**: [Download from FMA GitHub](https://github.com/mdeff/fma)
   - **AudioSet**: Use `download_audioset_subset.py` for music-focused clips
   - Extract to `./datasets/` directory

## Usage

### Run Advanced Framework (Week 4-5)
```bash
python adaptive_preprocessing_framework.py
```

Features:
- Spectral gating, HPSS, Spleeter, LUFS normalization
- Intelligent adaptive routing
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Before/after spectrogram visualization
- Automatic >4% improvement validation

### Run Phase 1 Baseline (Validated)
```bash
python adaptive_preprocessing_framework.py
```

**Validated Results**: 50.98% adaptive vs 45.22% baseline (+5.76% improvement)

### Download AudioSet Music Clips
```bash
python download_audioset_subset.py
```

Filters and downloads 12 music genres from AudioSet.

### Generate Analysis Charts
```bash
python analyze_results.py
```

Results saved to:
- `./results_advanced/` - Performance metrics and spectrograms
- `./results/` - Phase 1 baseline results

## Methodology

### Adaptive Preprocessing Pipeline

1. **Audio Loading**: Load audio clips at 22.05kHz
2. **Noise Characterization**: 
   - Analyze spectral flatness, centroid, zero-crossing rate
   - Classify as broadband_noise/harmonic_rich/vocal_heavy/general
3. **Intelligent Routing**: 
   - Broadband noise → Spectral gating + HPSS
   - Harmonic-rich → HPSS + LUFS normalization
   - Vocal-heavy → Spleeter separation (when enabled)
   - General → Lightweight spectral gating
4. **Feature Extraction**: Convert to 128×128 mel-spectrograms
5. **Classification**: 4-layer CNN with batch normalization

### CNN Architecture

- **Input**: 128×128 mel-spectrogram
- **Layers**: 4 convolutional blocks (32→64→128→256 filters, 3×3 kernels)
- **Regularization**: Batch normalization + dropout (0.25-0.5)
- **Pooling**: Max pooling (2×2) + Global average pooling
- **Output**: Softmax over 8 genre classes (FMA-medium)

### Dataset: FMA-medium

- **Size**: ~25,000 tracks, 8 genres
- **Genres**: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock
- **Training**: 22,930 samples
- **Validation**: Stratified split
- **Duration**: 29-second clips

## Results

### Phase 1 (Validated)
- **Baseline**: 45.22% accuracy (no preprocessing)
- **Adaptive**: 50.98% accuracy (classical methods)
- **Improvement**: +5.76% (exceeds >4% requirement)
- **Dataset**: Full FMA-medium (22,930 training samples)

### Week 4-5 (Advanced Methods)
- Implemented comprehensive framework with 6 preprocessing methods
- Added comprehensive evaluation metrics
- Added visualization tools
- Performance benchmarks in progress

### Performance Characteristics
- **Processing Time**: Lightweight adaptive routing <10x baseline
- **Memory**: Efficient batch processing
- **Scalability**: Tested on 2,000+ samples successfully

## Roadmap

### ✅ Phase 1 (Completed): Classical Denoising Baselines
- Spectral subtraction, median filtering, Wiener filtering
- Adaptive routing based on noise characteristics
- FMA-medium dataset integration
- Visualization and analysis tools
- **Validated: +5.76% improvement on full dataset**

### ✅ Week 4-5 (Completed): Advanced Preprocessing
- Spectral gating with smooth sigmoid function
- HPSS (Harmonic-Percussive Source Separation)
- Spleeter 2-stem vocal separation
- LUFS normalization (EBU R128)
- Demucs ML denoising integration (optional)
- Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
- Before/after spectrogram visualization
- Intelligent adaptive routing system

### 🔄 In Progress: Framework Optimization
- Restructure to single baseline + single adaptive comparison
- Dataset-wide noise analysis and statistics
- Selective method application based on noise distribution
- Performance optimization for large-scale experiments

### 📋 Planned: Comprehensive Evaluation
- Extended benchmarking on multiple datasets
- Ablation studies across all methods
- Processing time vs accuracy tradeoffs
- Final documentation and analysis

## Dependencies

### Core Requirements
- Python 3.9+
- TensorFlow 2.7+
- librosa 0.11+
- NumPy 1.22+
- SciPy 1.13+
- scikit-learn 1.1+
- matplotlib 3.5+
- seaborn 0.11+
- pandas 1.3+

### Advanced Preprocessing (requirements_advanced.txt)
- spleeter 2.4+ (vocal separation)
- pyloudnorm 0.2+ (LUFS normalization)
- demucs 4.0+ (ML denoising, optional - performance intensive)

See `requirements.txt` and `requirements_advanced.txt` for complete dependency lists.

## Technical Notes

### Performance Considerations
- **Demucs**: State-of-the-art quality but very slow (CPU intensive). Disabled by default for batch processing.
- **Spleeter**: Good quality, moderate speed. Requires pretrained models (~100MB download).
- **Lightweight Adaptive**: Uses spectral gating + HPSS for optimal speed/quality tradeoff.

### Known Issues
- LUFS normalization may cause artifacts with very low-energy audio (handled with NaN checks)
- Demucs not recommended for large-scale experiments (>1000 samples) without GPU acceleration

## Contributing

This is an active research project. Contributions, suggestions, and feedback are welcome via issues or pull requests.

## License

MIT License - feel free to use this code for research or educational purposes.

## Acknowledgments

- **FMA Dataset**: [https://github.com/mdeff/fma](https://github.com/mdeff/fma)
- **AudioSet**: [https://research.google.com/audioset/](https://research.google.com/audioset/)
- **Spleeter**: Deezer Research - [https://github.com/deezer/spleeter](https://github.com/deezer/spleeter)
- **Demucs**: Facebook Research - [https://github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)

## Citation

If you use this framework in your research, please cite:
```
Adaptive Noise-Aware Preprocessing for Music Genre Classification
Final Year Project, 2026
```

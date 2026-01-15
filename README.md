# Adaptive Noise-Aware Preprocessing for Music Genre Classification

A systematic framework for evaluating and comparing various preprocessing methods to improve music genre classification on noisy, real-world datasets.

## Overview

This project investigates how different denoising and preprocessing techniques affect the performance of music genre classification models. Phase 1 (current) focuses on classical signal processing methods with adaptive routing based on noise characteristics.

## Features

- **Classical Denoising Methods**:
  - Spectral Subtraction (stationary broadband noise)
  - Median Filtering (transient/impulsive noise)
  - Wiener Filtering (general adaptive noise reduction)

- **Adaptive Routing**: Automatically selects the most appropriate denoising method based on:
  - Spectral Flatness
  - Spectral Centroid
  - Zero-Crossing Rate

- **Visualization Tools**: Generate before/after spectrograms and performance comparison charts

- **Analysis & Reporting**: Automated generation of accuracy comparisons, per-genre F1 scores, noise distribution, and confusion matrices

## Project Structure

```
.
├── poc_fma_adaptive.py          # Main POC for FMA dataset
├── poc_mgc_adaptive.py          # POC for GTZAN dataset (synthetic noise)
├── poc_audioset_adaptive.py     # POC for AudioSet (not yet tested)
├── visualize_denoising.py       # Spectrogram visualization tool
├── analyze_results.py           # Results analysis and chart generation
├── test_functions.py            # Unit tests for denoising functions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/music-genre-preprocessing.git
cd music-genre-preprocessing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
   - **FMA-medium**: [Download from FMA GitHub](https://github.com/mdeff/fma)
   - **GTZAN**: Available on Kaggle
   - Extract to `./datasets/` directory

## Usage

### Run FMA POC
```bash
python poc_fma_adaptive.py
```

### Run GTZAN POC
```bash
python poc_mgc_adaptive.py
```

### Generate Visualizations
```bash
python visualize_denoising.py --dataset ./datasets/fma_medium --num-samples 5
```

Results will be saved to:
- `./results/` - Performance charts and summary
- `./visualizations/` - Before/after spectrograms

### Quick Validation
```bash
python test_functions.py
```

## Methodology

### Preprocessing Pipeline

1. **Audio Loading**: Load 29-second clips at 22.05kHz
2. **Noise Characterization**: Analyze spectral and temporal features
3. **Adaptive Denoising**: Route to appropriate denoiser
4. **Feature Extraction**: Convert to 128×128 mel-spectrograms
5. **Classification**: 4-layer CNN with batch normalization

### CNN Architecture

- **Input**: 128×128 mel-spectrogram
- **Layers**: 4 convolutional blocks (32→64→128→256 filters)
- **Regularization**: Batch normalization + dropout (0.25-0.5)
- **Pooling**: Global average pooling
- **Output**: Softmax over genre classes

## Results

Results vary by dataset and noise characteristics. The framework compares:
- **Baseline**: Direct feature extraction (no denoising)
- **Adaptive**: Noise-aware preprocessing

See `./results/` directory after running experiments for detailed performance metrics.

## Roadmap

### Phase 1 (Current): Classical Denoising Baselines
- ✅ Spectral subtraction, median filtering, Wiener filtering
- ✅ Adaptive routing based on noise characteristics
- ✅ FMA-medium and GTZAN integration
- ✅ Visualization and analysis tools

### Phase 2 (Weeks 4-7): ML-Based Denoising
- RNNoise integration (speech-trained baseline)
- Spectral gating implementation
- Spleeter for source separation
- Music-adapted RNNoise training

### Phase 3 (Weeks 8-9): Trainable Preprocessing
- PCEN (Per-Channel Energy Normalization) with learnable parameters
- End-to-end joint training (preprocessing + classifier)
- Ablation studies on PCEN parameters

### Phase 4 (Weeks 10-12): Comprehensive Benchmarking
- Evaluation on clean vs noisy datasets
- Ablation studies across all methods
- Final analysis and documentation

## Dependencies

- Python 3.8+
- TensorFlow 2.7+
- librosa 0.11+
- NumPy 1.22+
- SciPy 1.13+
- scikit-learn 1.1+
- matplotlib 3.5+
- seaborn 0.11+
- pandas 1.3+

See `requirements.txt` for full list.

## Contributing

This is a research project. Contributions, suggestions, and feedback are welcome via issues or pull requests.

## License

MIT License - feel free to use this code for research or educational purposes.

## Acknowledgments

- FMA Dataset: [https://github.com/mdeff/fma](https://github.com/mdeff/fma)
- GTZAN Dataset: [http://marsyas.info/downloads/datasets.html](http://marsyas.info/downloads/datasets.html)
- AudioSet: [https://research.google.com/audioset/](https://research.google.com/audioset/)


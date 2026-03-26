# Adaptive Preprocessing Framework for Music Genre Classification

An adaptive audio preprocessing framework that routes each audio sample to its most suitable preprocessing method before music genre classification, trained using oracle-guided meta-learning on FMA-Medium.

---

## Project Structure

```
├── adaptive_preprocessing_framework.py  # Core framework (preprocessing, routing, CNN)
├── benchmark_suite.py                   # Benchmarking engine (multi-run, CI, per-genre)
├── app_streamlit.py                     # Streamlit research interface
├── backend/
│   └── api.py                           # Flask + SocketIO API
├── frontend/
│   ├── src/                             # React source (Vite + Tailwind)
│   └── dist/                            # Production build
├── requirements.txt                     # All Python dependencies
└── datasets/                            # Place FMA-Medium and metadata here (not tracked)
```

---

## Prerequisites

- Python 3.9+
- Node.js 18+ (for frontend development)
- FMA-Medium dataset + metadata (`datasets/fma_medium/`, `datasets/fma_metadata/tracks.csv`)

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd POC

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# Install all dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Benchmark

```bash
python benchmark_suite.py                   # Full benchmark (3 runs, 5000 samples, 50 epochs)
python benchmark_suite.py --runs 5          # 5 runs for stronger statistical confidence
python benchmark_suite.py --quick           # Quick test (1 run, 500 samples, 10 epochs)
python benchmark_suite.py --snr 15          # Evaluate under noisy conditions (SNR=15 dB)
```

Results (CSV + JSON) are saved to `results/benchmarks/`.

### 2. Start the Flask Backend

```bash
cd backend
python api.py
# API available at http://localhost:5000
```

### 3. Start the Streamlit Interface

```bash
streamlit run app_streamlit.py
# Available at http://localhost:8501
```

### 4. Start the React Frontend

```bash
cd frontend
npm install
npm run dev       # Development server at http://localhost:5173
npm run build     # Build to frontend/dist/
```

> The React frontend connects to the Flask backend at `http://localhost:5000`.

---

## Methods Compared

| Method | Description |
|--------|-------------|
| No Preprocessing | Raw mel-spectrograms (baseline) |
| Rule-Based Adaptive | SNR + spectral flatness thresholds route to one of 6 methods |
| Model-Based Adaptive | MLP router (10→64→32→16→5) trained with oracle-guided labels |

---

## Key Results (FMA-Medium, 5 runs, 5000 samples, 50 epochs)

| Method | Accuracy | F1-Score | 95% CI (Accuracy) |
|--------|----------|----------|--------------------|
| No Preprocessing | 45.3% | 0.446 | ±0.021 |
| Rule-Based Adaptive | 44.6% | 0.434 | — |
| **Model-Based Adaptive** | **46.5%** | **0.454** | **±0.013** |

Model-based routing achieves a **35% reduction in confidence interval width** over baseline, indicating more consistent classification decisions.

---

## Dataset Setup

Download FMA-Medium from [https://github.com/mdeff/fma](https://github.com/mdeff/fma) and place as:

```
datasets/
├── fma_medium/          # Audio files (MP3), organised in subdirectories
└── fma_metadata/
    └── tracks.csv
```

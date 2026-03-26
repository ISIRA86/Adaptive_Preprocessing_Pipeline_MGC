"""
Benchmark suite for the Adaptive Audio Preprocessing Framework.

Usage:
    python benchmark_suite.py                    # Full benchmark (3 runs)
    python benchmark_suite.py --runs 5           # 5 runs for more rigour
    python benchmark_suite.py --quick            # Quick test (1 run, 500 samples, 10 epochs)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import from the main framework
from adaptive_preprocessing_framework import (
    load_fma_dataset,
    load_audioset_music_dataset,
    baseline_pipeline,
    adaptive_pipeline_granular,
    adaptive_pipeline_model_based,
    adaptive_pipeline_minimal,
    build_genre_cnn,
    get_training_callbacks,
    train_routing_model,
    calculate_comprehensive_metrics,
    _add_noise_at_snr_silent,
    SR, N_MELS, MEL_FRAMES, EPOCHS, BATCH_SIZE,
    TARGET_GENRES, AUDIOSET_TARGET_GENRES, RESULTS_DIR
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras


# ======================== CONFIGURATION ================================

BENCHMARK_DIR = os.path.join(RESULTS_DIR, 'benchmarks')
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# Random seeds for reproducibility
SEEDS = [42, 123, 456, 789, 1024]


# ======================== HELPER FUNCTIONS =============================

def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate mean and 95% confidence interval"""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h


def format_metric_with_ci(mean, ci_low, ci_high):
    """Format metric as 'mean ± ci'"""
    ci = (ci_high - ci_low) / 2
    return f"{mean:.4f} ± {ci:.4f}"


def print_section_header(text, width=70):
    """Print a formatted section header"""
    print('\n' + '=' * width)
    print(text.center(width))
    print('=' * width + '\n')


# ======================== BASELINE METHODS =============================

class BaselineMethod:
    """Base class for preprocessing methods"""
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def process(self, audio_batch):
        raise NotImplementedError


class NoPreprocessing(BaselineMethod):
    def __init__(self):
        super().__init__(
            "No Preprocessing",
            "Raw mel-spectrograms without any preprocessing"
        )
    
    def process(self, audio_batch):
        return baseline_pipeline(audio_batch), {}


class SingleMethodSpectralGating(BaselineMethod):
    def __init__(self):
        super().__init__(
            "Spectral Gating (Single)",
            "Same spectral gating applied to all samples"
        )
    
    def process(self, audio_batch):
        from adaptive_preprocessing_framework import denoise_spectral_gating, audio_to_multi_features
        X = []
        for audio in audio_batch:
            processed = denoise_spectral_gating(audio, threshold_db=-40, alpha=0.1)
            features = audio_to_multi_features(processed)
            X.append(features)
        return np.array(X)[..., np.newaxis], {"spectral_gating": len(audio_batch)}


class RandomPreprocessing(BaselineMethod):
    def __init__(self):
        super().__init__(
            "Random Selection",
            "Randomly apply preprocessing methods"
        )
    
    def process(self, audio_batch):
        from adaptive_preprocessing_framework import (
            denoise_spectral_gating, denoise_wiener_filter_gentle,
            audio_to_multi_features, METHOD_ID_TO_NAME, apply_preprocessing_method
        )
        X = []
        stats = defaultdict(int)
        
        for audio in audio_batch:
            method_id = np.random.randint(0, len(METHOD_ID_TO_NAME))
            processed = apply_preprocessing_method(audio, method_id)
            features = audio_to_multi_features(processed)
            X.append(features)
            stats[METHOD_ID_TO_NAME[method_id]] += 1
        
        return np.array(X)[..., np.newaxis], dict(stats)


class RuleBasedAdaptive(BaselineMethod):
    def __init__(self):
        super().__init__(
            "Rule-Based Adaptive",
            "Granular rule-based preprocessing selection"
        )
    
    def process(self, audio_batch):
        return adaptive_pipeline_granular(audio_batch)


class ModelBasedAdaptive(BaselineMethod):
    def __init__(self):
        super().__init__(
            "Model-Based Adaptive (Ours)",
            "Neural network meta-learning routing"
        )
    
    def process(self, audio_batch):
        return adaptive_pipeline_model_based(audio_batch)


class MinimalPreprocessing(BaselineMethod):
    def __init__(self):
        super().__init__(
            "Minimal Preprocessing",
            "Only preprocess very noisy samples"
        )
    
    def process(self, audio_batch):
        return adaptive_pipeline_minimal(audio_batch)


# ======================== BENCHMARKING ENGINE ==========================

class BenchmarkEngine:
    """Main benchmarking engine"""

    SUPPORTED_DATASETS = ('fma', 'audioset')

    def __init__(self, max_samples=1000, epochs=20, n_runs=3, verbose=True, dataset='fma', test_snr=0, routing_samples=500):
        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(f"dataset must be one of {self.SUPPORTED_DATASETS}, got '{dataset}'")
        self.max_samples = max_samples
        self.epochs = epochs
        self.n_runs = n_runs
        self.verbose = verbose
        self.dataset = dataset
        self.test_snr = test_snr
        self.routing_samples = routing_samples
        self.results = {}
        self.timestamp = f"{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def log(self, message):
        if self.verbose:
            print(message)

    def load_data(self):
        """Load and split dataset (FMA or AudioSet based on self.dataset)"""
        if self.dataset == 'audioset':
            self.log("Loading AudioSet music dataset...")
            X_audio, y, genres = load_audioset_music_dataset(max_samples=self.max_samples)
        else:
            self.log("Loading FMA dataset...")
            X_audio, y, genres = load_fma_dataset(max_samples=self.max_samples)

        self.genres = genres
        self.num_classes = len(genres)

        self.log(f"  Loaded {len(X_audio)} samples across {len(genres)} genres")
        self.log(f"  Genres: {', '.join(genres)}")

        return X_audio, y
    
    def run_single_experiment(self, X_train_audio, X_test_audio, y_train, y_test, 
                               method, seed, should_train_routing=False, 
                               baseline_model=None):
        """Run a single experiment with one method"""
        set_seeds(seed)
        
        # Train routing model if needed
        if should_train_routing and baseline_model is not None:
            # Cap oracle SNR to keep routing labels reliable during training
            oracle_snr = min(self.test_snr, 10) if self.test_snr > 0 else 0
            self.log(f"    Training routing model on {self.routing_samples} samples (oracle_snr={oracle_snr} dB, eval_snr={self.test_snr} dB)...")
            train_routing_model(
                X_train_audio, y_train,
                n_samples=self.routing_samples,
                classifier_model=baseline_model,
                epochs=self.epochs,
                noise_snr=oracle_snr
            )
        
        # Process audio
        self.log(f"    Processing {len(X_train_audio)} training samples...")
        start_time = time.time()
        X_train, train_stats = method.process(X_train_audio)
        X_test, test_stats = method.process(X_test_audio)
        processing_time = time.time() - start_time
        
        # Build and train model
        input_shape = X_train.shape[1:]
        model = build_genre_cnn(input_shape, self.num_classes)

        cw_values = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(cw_values))

        self.log(f"    Training CNN for {self.epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=BATCH_SIZE,
            class_weight=class_weights,
            callbacks=get_training_callbacks(),
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test, verbose=0)
        preds = np.argmax(y_pred_proba, axis=1)
        
        metrics = calculate_comprehensive_metrics(y_test, preds, y_pred_proba, self.genres)
        metrics['processing_time'] = processing_time
        metrics['preprocessing_stats'] = train_stats
        metrics['training_history'] = {
            'loss': history.history['loss'][-1],
            'val_loss': history.history.get('val_loss', [0])[-1]
        }
        
        return metrics, model
    
    def run_benchmark(self, methods_to_test=None):
        """Run full benchmark suite"""
        print_section_header("ADAPTIVE PREPROCESSING BENCHMARK SUITE")
        
        # Core comparison: No Preprocessing (baseline), Rule-Based, Model-Based (ours)
        if methods_to_test is None:
            methods_to_test = [
                NoPreprocessing(),
                RuleBasedAdaptive(),
                ModelBasedAdaptive()
            ]
        
        # Load data once
        X_audio, y = self.load_data()
        
        # Results storage
        all_results = {method.name: [] for method in methods_to_test}
        
        for run_idx in range(self.n_runs):
            seed = SEEDS[run_idx % len(SEEDS)]
            set_seeds(seed)
            
            print_section_header(f"RUN {run_idx + 1}/{self.n_runs} (seed={seed})")
            
            # Split data
            X_train_audio, X_test_audio, y_train, y_test = train_test_split(
                X_audio, y, test_size=0.2, stratify=y, random_state=seed
            )

            # Apply test noise once — same degraded audio for all methods
            if self.test_snr > 0:
                self.log(f"  Applying test noise at SNR={self.test_snr} dB...")
                X_test_audio = [_add_noise_at_snr_silent(a, target_snr_db=self.test_snr)
                                for a in X_test_audio]
            
            # First, train baseline for routing model training
            self.log("Training baseline model for routing...")
            X_train_baseline, _ = NoPreprocessing().process(X_train_audio)
            X_test_baseline, _ = NoPreprocessing().process(X_test_audio)
            input_shape = X_train_baseline.shape[1:]
            baseline_model = build_genre_cnn(input_shape, self.num_classes)
            cw_values = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights_baseline = dict(enumerate(cw_values))
            baseline_model.fit(
                X_train_baseline, y_train,
                validation_split=0.15,
                epochs=self.epochs,
                batch_size=BATCH_SIZE,
                class_weight=class_weights_baseline,
                callbacks=get_training_callbacks(),
                verbose=0
            )
            
            # Run each method
            for method in methods_to_test:
                self.log(f"\n  Testing: {method.name}")
                self.log(f"    {method.description}")
                
                # Check if model-based needs routing model training
                needs_routing = isinstance(method, ModelBasedAdaptive)
                
                metrics, _ = self.run_single_experiment(
                    X_train_audio, X_test_audio, y_train, y_test,
                    method, seed,
                    should_train_routing=needs_routing,
                    baseline_model=baseline_model if needs_routing else None
                )
                
                all_results[method.name].append(metrics)
                
                self.log(f"    Accuracy: {metrics['accuracy']:.4f}")
                self.log(f"    F1-Score: {metrics['f1_macro']:.4f}")
        
        self.results = all_results
        return all_results
    
    def generate_summary_table(self):
        """Generate summary statistics table"""
        print_section_header("BENCHMARK RESULTS SUMMARY")
        
        summary_data = []
        
        for method_name, runs in self.results.items():
            # Extract metrics from all runs
            accuracies = [r['accuracy'] for r in runs]
            f1_scores = [r['f1_macro'] for r in runs]
            auc_scores = [r['auc_roc_macro'] for r in runs]
            times = [r['processing_time'] for r in runs]
            
            # Calculate statistics
            acc_mean, acc_low, acc_high = calculate_confidence_interval(accuracies)
            f1_mean, f1_low, f1_high = calculate_confidence_interval(f1_scores)
            auc_mean, auc_low, auc_high = calculate_confidence_interval(auc_scores)
            time_mean = np.mean(times)
            
            summary_data.append({
                'Method': method_name,
                'Accuracy': format_metric_with_ci(acc_mean, acc_low, acc_high),
                'Accuracy_mean': acc_mean,
                'F1-Score': format_metric_with_ci(f1_mean, f1_low, f1_high),
                'F1_mean': f1_mean,
                'AUC-ROC': format_metric_with_ci(auc_mean, auc_low, auc_high),
                'AUC_mean': auc_mean,
                'Time (s)': f"{time_mean:.1f}",
                'Time_mean': time_mean
            })
        
        # Sort by F1-Score
        summary_data.sort(key=lambda x: x['F1_mean'], reverse=True)
        
        # Print table
        df = pd.DataFrame(summary_data)
        print_cols = ['Method', 'Accuracy', 'F1-Score', 'AUC-ROC', 'Time (s)']
        print(df[print_cols].to_string(index=False))
        
        # Calculate improvements over baseline
        baseline_f1 = None
        for row in summary_data:
            if row['Method'] == 'No Preprocessing':
                baseline_f1 = row['F1_mean']
                break
        
        if baseline_f1:
            print("\n" + "-" * 70)
            print("IMPROVEMENT OVER BASELINE:")
            print("-" * 70)
            for row in summary_data:
                if row['Method'] != 'No Preprocessing':
                    improvement = ((row['F1_mean'] - baseline_f1) / baseline_f1) * 100
                    print(f"  {row['Method']:<30}: {improvement:+.2f}%")
        
        return df
    
    def generate_per_genre_analysis(self):
        """Generate per-genre performance analysis"""
        print_section_header("PER-GENRE ANALYSIS")
        
        genre_data = []
        
        for genre in self.genres:
            row = {'Genre': genre}
            for method_name, runs in self.results.items():
                f1_scores = [r['per_class'][genre]['f1'] for r in runs]
                mean_f1, _, _ = calculate_confidence_interval(f1_scores)
                row[method_name] = mean_f1
            genre_data.append(row)
        
        df = pd.DataFrame(genre_data)
        print(df.to_string(index=False))
        
        return df
    
    def generate_latex_table(self):
        """Generate LaTeX formatted table for paper"""
        print_section_header("LATEX TABLE")
        
        latex = r"""
\begin{table}[h]
\centering
\caption{Comparison of Preprocessing Methods for Music Genre Classification}
\label{tab:benchmark_results}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{AUC-ROC} \\
\midrule
"""
        
        # Sort by F1-Score
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: np.mean([r['f1_macro'] for r in x[1]]),
            reverse=True
        )
        
        for method_name, runs in sorted_results:
            accuracies = [r['accuracy'] for r in runs]
            f1_scores = [r['f1_macro'] for r in runs]
            auc_scores = [r['auc_roc_macro'] for r in runs]
            
            acc_mean, acc_low, acc_high = calculate_confidence_interval(accuracies)
            f1_mean, f1_low, f1_high = calculate_confidence_interval(f1_scores)
            auc_mean, auc_low, auc_high = calculate_confidence_interval(auc_scores)
            
            # Bold the best method (our approach)
            if "Model-Based" in method_name:
                latex += f"\\textbf{{{method_name}}} & "
                latex += f"\\textbf{{{acc_mean:.3f} $\\pm$ {(acc_high-acc_low)/2:.3f}}} & "
                latex += f"\\textbf{{{f1_mean:.3f} $\\pm$ {(f1_high-f1_low)/2:.3f}}} & "
                latex += f"\\textbf{{{auc_mean:.3f} $\\pm$ {(auc_high-auc_low)/2:.3f}}} \\\\\n"
            else:
                latex += f"{method_name} & "
                latex += f"{acc_mean:.3f} $\\pm$ {(acc_high-acc_low)/2:.3f} & "
                latex += f"{f1_mean:.3f} $\\pm$ {(f1_high-f1_low)/2:.3f} & "
                latex += f"{auc_mean:.3f} $\\pm$ {(auc_high-auc_low)/2:.3f} \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        print(latex)
        return latex
    
    def generate_visualizations(self):
        """Generate publication-quality visualizations"""
        print_section_header("GENERATING VISUALIZATIONS")
        
        # Set style (use seaborn-whitegrid for matplotlib 3.7+)
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            plt.style.use('whitegrid')
        colors = sns.color_palette("husl", len(self.results))
        
        # 1. Bar chart comparing methods
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics_to_plot = [
            ('accuracy', 'Accuracy'),
            ('f1_macro', 'F1-Score (Macro)'),
            ('auc_roc_macro', 'AUC-ROC (Macro)')
        ]
        
        method_names = list(self.results.keys())
        x = np.arange(len(method_names))
        
        for ax_idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
            means = []
            errors = []
            
            for method_name in method_names:
                values = [r[metric_key] for r in self.results[method_name]]
                mean = np.mean(values)
                ci = stats.sem(values) * stats.t.ppf(0.975, len(values) - 1)
                means.append(mean)
                errors.append(ci)
            
            bars = axes[ax_idx].bar(x, means, yerr=errors, capsize=5, 
                                     color=colors, edgecolor='black', linewidth=1)
            
            axes[ax_idx].set_xlabel('Method')
            axes[ax_idx].set_ylabel(metric_name)
            axes[ax_idx].set_title(f'{metric_name} Comparison')
            axes[ax_idx].set_xticks(x)
            axes[ax_idx].set_xticklabels([n.replace(' ', '\n') for n in method_names], 
                                          rotation=45, ha='right', fontsize=8)
            axes[ax_idx].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                axes[ax_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                   f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        fig.savefig(os.path.join(BENCHMARK_DIR, f'comparison_bars_{self.timestamp}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Radar chart for per-genre analysis
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(self.genres), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for idx, (method_name, runs) in enumerate(self.results.items()):
            values = [np.mean([r['per_class'][g]['f1'] for r in runs]) for g in self.genres]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=method_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.genres, fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Per-Genre F1-Score Comparison', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(os.path.join(BENCHMARK_DIR, f'radar_chart_{self.timestamp}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Box plot showing variance
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data_for_boxplot = []
        labels = []
        for method_name, runs in self.results.items():
            data_for_boxplot.append([r['f1_macro'] for r in runs])
            labels.append(method_name)
        
        bp = ax.boxplot(data_for_boxplot, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=9)
        ax.set_ylabel('F1-Score (Macro)')
        ax.set_title('F1-Score Distribution Across Runs')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        fig.savefig(os.path.join(BENCHMARK_DIR, f'boxplot_{self.timestamp}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"  Visualizations saved to: {BENCHMARK_DIR}")
    
    def export_results(self, stat_results=None):
        """Export results to CSV and JSON"""
        print_section_header("EXPORTING RESULTS")
        
        # CSV export
        csv_data = []
        for method_name, runs in self.results.items():
            for run_idx, metrics in enumerate(runs):
                row = {
                    'method': method_name,
                    'run': run_idx + 1,
                    'accuracy': metrics['accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'f1_weighted': metrics['f1_weighted'],
                    'precision_weighted': metrics['precision_weighted'],
                    'recall_weighted': metrics['recall_weighted'],
                    'auc_roc_macro': metrics['auc_roc_macro'],
                    'processing_time': metrics['processing_time']
                }
                # Add per-genre F1 scores
                for genre in self.genres:
                    row[f'f1_{genre}'] = metrics['per_class'][genre]['f1']
                csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(BENCHMARK_DIR, f'benchmark_results_{self.timestamp}.csv')
        df.to_csv(csv_path, index=False)
        self.log(f"  CSV saved to: {csv_path}")
        
        # JSON export (full results)
        json_data = {
            'timestamp': self.timestamp,
            'config': {
                'max_samples': self.max_samples,
                'epochs': self.epochs,
                'n_runs': self.n_runs,
                'genres': self.genres
            },
            'results': {}
        }
        
        for method_name, runs in self.results.items():
            json_data['results'][method_name] = {
                'runs': [],
                'summary': {}
            }
            
            accuracies = []
            f1_scores = []
            
            for metrics in runs:
                # Convert numpy types to native Python types for JSON
                run_data = {
                    'accuracy': float(metrics['accuracy']),
                    'f1_macro': float(metrics['f1_macro']),
                    'auc_roc_macro': float(metrics['auc_roc_macro']),
                    'processing_time': float(metrics['processing_time'])
                }
                json_data['results'][method_name]['runs'].append(run_data)
                accuracies.append(metrics['accuracy'])
                f1_scores.append(metrics['f1_macro'])
            
            # Summary statistics
            acc_mean, acc_low, acc_high = calculate_confidence_interval(accuracies)
            f1_mean, f1_low, f1_high = calculate_confidence_interval(f1_scores)
            
            json_data['results'][method_name]['summary'] = {
                'accuracy_mean': float(acc_mean),
                'accuracy_ci': [float(acc_low), float(acc_high)],
                'f1_mean': float(f1_mean),
                'f1_ci': [float(f1_low), float(f1_high)]
            }
        
        json_path = os.path.join(BENCHMARK_DIR, f'benchmark_results_{self.timestamp}.json')
        
        # Include statistical tests if available
        if stat_results:
            json_data['statistical_tests'] = stat_results
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        self.log(f"  JSON saved to: {json_path}")
        
        return csv_path, json_path
    
    def run_full_benchmark(self):
        """Run the complete benchmark pipeline"""
        start_time = time.time()
        
        # Run experiments
        self.run_benchmark()
        
        # Generate outputs
        summary_df = self.generate_summary_table()
        genre_df = self.generate_per_genre_analysis()
        latex = self.generate_latex_table()
        self.generate_visualizations()
        
        # Statistical tests (need ≥ 2 runs)
        stat_results = None
        if self.n_runs >= 2:
            stat_results = run_statistical_tests(self.results)
        
        csv_path, json_path = self.export_results(stat_results=stat_results)
        
        total_time = time.time() - start_time
        
        print_section_header("BENCHMARK COMPLETE")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"\nOutputs saved to: {BENCHMARK_DIR}")
        print(f"  - CSV:    {os.path.basename(csv_path)}")
        print(f"  - JSON:   {os.path.basename(json_path)}")
        print(f"  - Charts: comparison_bars_*.png, radar_chart_*.png, boxplot_*.png")
        
        return self.results


# ======================== STATISTICAL TESTS ============================

def run_statistical_tests(results):
    """Run statistical significance tests: Model-Based Adaptive vs No Preprocessing baseline."""
    print_section_header("STATISTICAL SIGNIFICANCE TESTS")

    model_based_key = 'Model-Based Adaptive (Ours)'
    baseline_key = 'No Preprocessing'

    model_f1  = [r['f1_macro']  for r in results.get(model_based_key, [])]
    model_acc = [r['accuracy']  for r in results.get(model_based_key, [])]
    base_f1   = [r['f1_macro']  for r in results.get(baseline_key, [])]
    base_acc  = [r['accuracy']  for r in results.get(baseline_key, [])]

    if len(model_f1) < 2 or len(base_f1) < 2:
        print("Not enough runs for statistical testing (need at least 2)")
        return {}

    print(f"\n--- {model_based_key} vs {baseline_key} ---")

    # Paired t-test on F1
    t_stat, p_value = stats.ttest_rel(model_f1, base_f1)
    print(f"  Paired t-test (F1):   t={t_stat:.4f},  p={p_value:.6f}", end='')
    print("  *" if p_value < 0.05 else "")

    # Wilcoxon signed-rank (non-parametric alternative)
    try:
        w_stat, w_p = stats.wilcoxon(model_f1, base_f1)
        print(f"  Wilcoxon signed-rank: W={w_stat:.2f}, p={w_p:.6f}", end='')
        print("  *" if w_p < 0.05 else "")
    except ValueError:
        w_stat, w_p = None, None
        print("  Wilcoxon: N/A (identical values)")

    # Effect size – Cohen's d
    diff = np.array(model_f1) - np.array(base_f1)
    cohens_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)
    effect_label = ('negligible' if abs(cohens_d) < 0.2
                    else 'small' if abs(cohens_d) < 0.5
                    else 'medium' if abs(cohens_d) < 0.8
                    else 'large')
    print(f"  Cohen's d = {cohens_d:.4f} ({effect_label})")

    # Accuracy and F1 deltas
    acc_diff = np.mean(model_acc) - np.mean(base_acc)
    f1_diff  = np.mean(model_f1)  - np.mean(base_f1)
    print(f"  Mean accuracy delta: {acc_diff:+.4f}")
    print(f"  Mean F1 delta:       {f1_diff:+.4f}")

    stat_results = {
        baseline_key: {
            't_statistic': float(t_stat),
            'p_value_ttest': float(p_value),
            'wilcoxon_W': float(w_stat) if w_stat is not None else None,
            'p_value_wilcoxon': float(w_p) if w_p is not None else None,
            'cohens_d': float(cohens_d),
            'effect_size': effect_label,
            'accuracy_delta': float(acc_diff),
            'f1_delta': float(f1_diff),
        }
    }

    return stat_results


# ======================== MAIN =========================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark Adaptive Audio Preprocessing')
    parser.add_argument('--runs', type=int, default=3, help='Number of experiment runs')
    parser.add_argument('--samples', type=int, default=1000, help='Max samples to load')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (1 run, 500 samples)')
    parser.add_argument('--snr', type=int, default=0,
                        help='Apply noise to test set at this SNR (dB). 0 = clean (default).')
    parser.add_argument('--routing-samples', type=int, default=500, dest='routing_samples',
                        help='Number of samples used to train the routing model (default: 500).')
    parser.add_argument('--dataset', choices=['fma', 'audioset', 'both'],
                        default='fma', help='Dataset to benchmark on (default: fma)')
    args = parser.parse_args()

    # Quick mode for testing
    if args.quick:
        args.runs = 1
        args.samples = 500
        args.epochs = 10
        print("QUICK MODE: 1 run, 500 samples, 10 epochs")

    datasets_to_run = ['fma', 'audioset'] if args.dataset == 'both' else [args.dataset]

    all_dataset_results = {}
    for ds in datasets_to_run:
        print(f"\n{'='*70}")
        print(f"  DATASET: {ds.upper()}")
        print(f"{'='*70}")

        # Create and run benchmark
        engine = BenchmarkEngine(
            max_samples=args.samples,
            epochs=args.epochs,
            n_runs=args.runs,
            verbose=True,
            dataset=ds,
            test_snr=args.snr,
            routing_samples=args.routing_samples
        )
        results = engine.run_full_benchmark()
        all_dataset_results[ds] = results

    if args.dataset == 'both' and len(all_dataset_results) == 2:
        print_section_header("CROSS-DATASET COMPARISON")
        print("F1-Score (Macro) across datasets:")
        print(f"  {'Method':<32} {'FMA':>10} {'AudioSet':>10} {'Delta':>10}")
        print("  " + "-" * 65)
        all_methods = set(all_dataset_results['fma'].keys())
        for method in sorted(all_methods):
            fma_runs   = all_dataset_results['fma'].get(method, [])
            aset_runs  = all_dataset_results['audioset'].get(method, [])
            if not fma_runs or not aset_runs:
                continue
            fma_f1  = np.mean([r['f1_macro'] for r in fma_runs])
            aset_f1 = np.mean([r['f1_macro'] for r in aset_runs])
            delta   = aset_f1 - fma_f1
            print(f"  {method:<32} {fma_f1:>10.4f} {aset_f1:>10.4f} {delta:>+10.4f}")

    # Skip duplicate loop — already handled above
    print("\n" + "=" * 70)
    print("BENCHMARK FINISHED SUCCESSFULLY")
    print("=" * 70)


if __name__ == '__main__':
    main()

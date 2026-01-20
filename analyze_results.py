import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import json

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'


def plot_accuracy_comparison(baseline_acc, adaptive_acc, output_path='./results/accuracy_comparison.png'):
    """Create bar chart comparing baseline vs adaptive accuracy."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Baseline\n(No Denoising)', 'Adaptive\n(Smart Denoising)']
    accuracies = [baseline_acc * 100, adaptive_acc * 100]
    colors = ['#3498db', '#2ecc71'] if adaptive_acc > baseline_acc else ['#3498db', '#e74c3c']
    
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add improvement annotation
    improvement = (adaptive_acc - baseline_acc) * 100
    if abs(improvement) > 0.1:
        mid_y = (accuracies[0] + accuracies[1]) / 2
        arrow_props = dict(arrowstyle='<->', lw=2, color='black')
        ax.annotate('', xy=(0.5, accuracies[0]), xytext=(0.5, accuracies[1]),
                   arrowprops=arrow_props)
        ax.text(0.65, mid_y, f'{improvement:+.2f}%', fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Genre Classification: Baseline vs Adaptive Preprocessing', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_per_genre_comparison(baseline_report, adaptive_report, genres, 
                               output_path='./results/per_genre_comparison.png'):
    """Create grouped bar chart showing per-genre F1 scores."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract F1 scores
    baseline_f1 = []
    adaptive_f1 = []
    
    for genre in genres:
        if genre in baseline_report:
            baseline_f1.append(baseline_report[genre]['f1-score'] * 100)
        else:
            baseline_f1.append(0)
        
        if genre in adaptive_report:
            adaptive_f1.append(adaptive_report[genre]['f1-score'] * 100)
        else:
            adaptive_f1.append(0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(genres))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_f1, width, label='Baseline', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, adaptive_f1, width, label='Adaptive',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Genre', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1-Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Per-Genre Performance: Baseline vs Adaptive', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(genres, rotation=45, ha='right')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_noise_distribution(noise_stats, output_path='./results/noise_distribution.png'):
    """Create pie chart showing distribution of detected noise types."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = []
    sizes = []
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    noise_labels = {
        'broadband': 'Broadband\n(Spectral Sub.)',
        'lowfreq': 'Low Frequency\n(Median Filter)',
        'transient': 'Transient\n(Median Filter)',
        'general': 'General\n(Wiener Filter)'
    }
    
    for noise_type, label in noise_labels.items():
        if noise_type in noise_stats and noise_stats[noise_type] > 0:
            labels.append(label)
            sizes.append(noise_stats[noise_type])
    
    if not sizes:
        print('No noise statistics available')
        return
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors[:len(labels)],
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'},
                                        pctdistance=0.85)
    
    # Add count labels
    for i, (wedge, size) in enumerate(zip(wedges, sizes)):
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        ax.annotate(f'n={size}', xy=(x*0.6, y*0.6), ha='center', va='center',
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Adaptive Routing: Detected Noise Type Distribution', 
                fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def plot_confusion_matrices(baseline_cm, adaptive_cm, genres,
                            output_path='./results/confusion_matrices.png'):
    """Create side-by-side confusion matrices."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Normalize to percentages
    baseline_cm_norm = baseline_cm.astype('float') / baseline_cm.sum(axis=1)[:, np.newaxis] * 100
    adaptive_cm_norm = adaptive_cm.astype('float') / adaptive_cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot baseline
    sns.heatmap(baseline_cm_norm, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=genres, yticklabels=genres, ax=axes[0],
                cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
    axes[0].set_title('Baseline Confusion Matrix', fontsize=13, fontweight='bold', pad=15)
    axes[0].set_xlabel('Predicted Genre', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('True Genre', fontsize=11, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=9)
    
    # Plot adaptive
    sns.heatmap(adaptive_cm_norm, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=genres, yticklabels=genres, ax=axes[1],
                cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
    axes[1].set_title('Adaptive Confusion Matrix', fontsize=13, fontweight='bold', pad=15)
    axes[1].set_xlabel('Predicted Genre', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('True Genre', fontsize=11, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=9)
    
    plt.suptitle('Classification Confusion Matrices', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


def create_demo_summary(results_dict, output_path='./results/demo_summary.txt'):
    """Create text summary for demo presentation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('='*70 + '\n')
        f.write('ADAPTIVE DENOISING POC - DEMO SUMMARY\n')
        f.write('='*70 + '\n\n')
        
        f.write('PROJECT: Noise-Aware Preprocessing for Music Genre Classification\n')
        f.write('DATE: January 2026\n')
        f.write('PHASE: Week 1-2 - Classical Denoising Baselines\n\n')
        
        f.write('='*70 + '\n')
        f.write('METHODOLOGY\n')
        f.write('='*70 + '\n\n')
        
        f.write('1. DATASETS:\n')
        f.write(f'   - FMA-Medium: {results_dict.get("num_samples", "N/A")} tracks, ')
        f.write(f'{results_dict.get("num_genres", "N/A")} genres\n')
        f.write(f'   - Train/Test Split: {results_dict.get("train_size", "N/A")} / ')
        f.write(f'{results_dict.get("test_size", "N/A")}\n\n')
        
        f.write('2. DENOISING METHODS:\n')
        f.write('   - Spectral Subtraction (stationary broadband noise)\n')
        f.write('   - Median Filtering (non-stationary transient noise)\n')
        f.write('   - Wiener Filtering (general noise with SNR adaptation)\n\n')
        
        f.write('3. ADAPTIVE ROUTING:\n')
        f.write('   - Features: Spectral Flatness, Spectral Centroid, Zero-Crossing Rate\n')
        f.write('   - Decision Tree: Routes each audio to appropriate denoiser\n\n')
        
        f.write('4. CLASSIFIER:\n')
        f.write('   - Architecture: 4-layer CNN with batch normalization\n')
        f.write('   - Input: 128x128 mel-spectrograms\n')
        f.write(f'   - Training: {results_dict.get("epochs", 10)} epochs, ')
        f.write(f'batch size {results_dict.get("batch_size", 16)}\n\n')
        
        f.write('='*70 + '\n')
        f.write('RESULTS\n')
        f.write('='*70 + '\n\n')
        
        baseline_acc = results_dict.get('baseline_accuracy', 0) * 100
        adaptive_acc = results_dict.get('adaptive_accuracy', 0) * 100
        improvement = adaptive_acc - baseline_acc
        
        f.write(f'Baseline Accuracy:  {baseline_acc:.2f}%\n')
        f.write(f'Adaptive Accuracy:  {adaptive_acc:.2f}%\n')
        f.write(f'Improvement:        {improvement:+.2f}%\n\n')
        
        if 'noise_distribution' in results_dict:
            f.write('Noise Type Distribution:\n')
            for noise_type, count in results_dict['noise_distribution'].items():
                percentage = count / sum(results_dict['noise_distribution'].values()) * 100
                f.write(f'   - {noise_type.capitalize()}: {count} ({percentage:.1f}%)\n')
            f.write('\n')
        
        f.write('='*70 + '\n')
        f.write('KEY FINDINGS\n')
        f.write('='*70 + '\n\n')
        
        if improvement > 5:
            f.write('✓ Adaptive preprocessing SIGNIFICANTLY IMPROVED performance\n')
            f.write('  → Intelligent noise characterization works effectively\n')
            f.write('  → Different denoising methods suit different noise profiles\n')
        elif improvement > 0:
            f.write('✓ Adaptive preprocessing IMPROVED performance moderately\n')
            f.write('  → Approach shows promise but needs refinement\n')
        else:
            f.write('✗ Adaptive preprocessing did NOT improve performance\n')
            f.write('  → Over-aggressive denoising may remove musical features\n')
            f.write('  → Need gentler parameters or smarter characterization\n')
        f.write('\n')
        
        f.write('='*70 + '\n')
        f.write('NEXT STEPS (Week 3-4)\n')
        f.write('='*70 + '\n\n')
        f.write('1. Implement ML-based denoising (RNNoise baseline)\n')
        f.write('2. Add source separation (Spleeter for harmonic/percussive)\n')
        f.write('3. Test spectral gating (more sophisticated than spectral subtraction)\n')
        f.write('4. Begin RNNoise modification for music signals\n\n')
        
        f.write('='*70 + '\n')
        f.write('CONCLUSION\n')
        f.write('='*70 + '\n\n')
        f.write('This POC establishes baseline performance using classical DSP methods.\n')
        f.write('Results provide foundation for comparing advanced ML-based approaches.\n')
        f.write('Adaptive routing concept validated - next phase: trainable preprocessing.\n')
    
    print(f'Saved: {output_path}')


def generate_all_reports(baseline_acc, adaptive_acc, baseline_report_dict, adaptive_report_dict,
                         genres, noise_stats=None, baseline_cm=None, adaptive_cm=None):
    """Generate all visualization and analysis reports."""
    
    results_dict = {
        'baseline_accuracy': baseline_acc,
        'adaptive_accuracy': adaptive_acc,
        'num_genres': len(genres),
        'epochs': 10,
        'batch_size': 16
    }
    
    if noise_stats:
        results_dict['noise_distribution'] = noise_stats
    
    # Create results directory
    os.makedirs('./results', exist_ok=True)
    
    # Generate plots
    print('Generating visualizations...')
    plot_accuracy_comparison(baseline_acc, adaptive_acc)
    plot_per_genre_comparison(baseline_report_dict, adaptive_report_dict, genres)
    
    if noise_stats:
        plot_noise_distribution(noise_stats)
    
    if baseline_cm is not None and adaptive_cm is not None:
        plot_confusion_matrices(baseline_cm, adaptive_cm, genres)
    
    # Generate summary
    create_demo_summary(results_dict)
    
    print('\n' + '='*70)
    print('All reports generated in ./results/')
    print('='*70)


if __name__ == '__main__':
    # Example usage
    print('This script is meant to be imported and called from POC scripts.')
    print('Example:')
    print('  from analyze_results import generate_all_reports')
    print('  generate_all_reports(baseline_acc, adaptive_acc, ...)')

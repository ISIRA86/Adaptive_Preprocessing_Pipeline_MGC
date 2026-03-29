from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import sys
import os
import json
import base64
import io
import threading
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_preprocessing_framework import (
    load_fma_dataset,
    baseline_pipeline,
    adaptive_pipeline_model_based,
    adaptive_pipeline_granular,
    train_routing_model,
    extract_audio_features_for_routing,
    apply_preprocessing_method,
    analyze_dataset_noise,
    build_genre_cnn,
    calculate_comprehensive_metrics,
    generate_spectrogram_figure,
    save_spectrogram_comparison,
    _add_noise_at_snr_silent,
    METHOD_ID_TO_NAME,
    ROUTING_MODEL,
    TARGET_GENRES,
    SR, N_MELS, RESULTS_DIR, SPECTROGRAMS_DIR
)

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state for experiment tracking
experiment_state = {
    'status': 'idle',  # idle, loading, analyzing, training_baseline, training_routing, training_adaptive, complete, error
    'progress': 0,
    'current_step': '',
    'logs': [],
    'results': None,
    'dataset': None,
    'models': {
        'baseline': None,
        'adaptive': None,
        'routing': None
    }
}

def log(message, level='INFO'):
    """Print formatted log message to console"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    prefix = {
        'INFO': '\033[94m[INFO]\033[0m',
        'SUCCESS': '\033[92m[SUCCESS]\033[0m',
        'WARNING': '\033[93m[WARNING]\033[0m',
        'ERROR': '\033[91m[ERROR]\033[0m',
        'STEP': '\033[95m[STEP]\033[0m',
    }.get(level, '[INFO]')
    print(f"{timestamp} {prefix} {message}", flush=True)

def emit_progress(status, progress, message, data=None):
    """Emit progress update via WebSocket and print to console"""
    experiment_state['status'] = status
    experiment_state['progress'] = progress
    experiment_state['current_step'] = message
    experiment_state['logs'].append({
        'timestamp': datetime.now().isoformat(),
        'message': message
    })
    
    # Print to console with formatting
    if status == 'error':
        log(message, 'ERROR')
    elif status == 'complete':
        log(message, 'SUCCESS')
    elif 'training' in status:
        log(f"[{progress}%] {message}", 'STEP')
    else:
        log(f"[{progress}%] {message}", 'INFO')
    
    socketio.emit('progress', {
        'status': status,
        'progress': progress,
        'message': message,
        'data': data
    })


# ==================== REST Endpoints ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '2.0.0',
        'features': {
            'model_based_routing': True,
            'preprocessing_methods': len(METHOD_ID_TO_NAME),
            'genres': len(TARGET_GENRES)
        }
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration"""
    return jsonify({
        'preprocessing_methods': METHOD_ID_TO_NAME,
        'genres': TARGET_GENRES,
        'sample_rate': SR,
        'mel_bins': N_MELS,
        'routing_features': [
            'SNR (dB)', 'Spectral Flatness (mean)', 'Spectral Flatness (std)',
            'Spectral Centroid', 'Zero Crossing Rate', 'Harmonic Ratio',
            'Percussive Ratio', 'RMS Energy (mean)', 'RMS Energy (std)',
            'Spectral Bandwidth'
        ]
    })


@app.route('/api/experiment/status', methods=['GET'])
def get_experiment_status():
    """Get current experiment status"""
    return jsonify({
        'status': experiment_state['status'],
        'progress': experiment_state['progress'],
        'current_step': experiment_state['current_step'],
        'logs': experiment_state['logs'][-50:],  # Last 50 logs
        'has_results': experiment_state['results'] is not None
    })


@app.route('/api/experiment/results', methods=['GET'])
def get_experiment_results():
    """Get experiment results"""
    if experiment_state['results'] is None:
        return jsonify({'error': 'No results available'}), 404
    return jsonify(experiment_state['results'])


@app.route('/api/experiment/start', methods=['POST'])
def start_experiment():
    """Start a new experiment"""
    if experiment_state['status'] not in ['idle', 'complete', 'error']:
        return jsonify({'error': 'Experiment already running'}), 400
    
    # Get parameters from request
    params = request.json or {}
    max_samples = params.get('max_samples', 1000)
    epochs = params.get('epochs', 20)
    routing_samples = params.get('routing_samples', 500)
    test_noise_snr = params.get('test_noise_snr', 0)
    
    # Reset state
    experiment_state['status'] = 'starting'
    experiment_state['progress'] = 0
    experiment_state['logs'] = []
    experiment_state['results'] = None
    
    # Run experiment in background thread
    thread = threading.Thread(
        target=run_experiment_async,
        args=(max_samples, epochs, routing_samples, test_noise_snr)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Experiment started', 'params': params})


def run_experiment_async(max_samples, epochs, routing_samples, test_noise_snr=0):
    """Run experiment asynchronously with progress updates"""
    global experiment_state

    # Seed for reproducibility across runs
    np.random.seed(42)

    try:
        print("\n" + "="*70)
        print("         ADAPTIVE AUDIO PREPROCESSING EXPERIMENT")
        print("="*70)
        print(f"Parameters: max_samples={max_samples}, epochs={epochs}, routing_samples={routing_samples}, test_noise_snr={test_noise_snr}")
        print("="*70 + "\n")
        
        # ========== STEP 1: Load Dataset ==========
        log("Starting dataset loading...", "STEP")
        emit_progress('loading', 5, 'Loading FMA dataset...')
        
        X_audio, y, genres = load_fma_dataset(max_samples=max_samples)
        log(f"Loaded {len(X_audio)} audio samples", "SUCCESS")
        
        log("Splitting into train/test sets...", "INFO")
        X_train_audio, X_test_audio, y_train, y_test = train_test_split(
            X_audio, y, test_size=0.2, stratify=y, random_state=42
        )
        log(f"Train: {len(X_train_audio)} samples, Test: {len(X_test_audio)} samples", "INFO")

        # Apply test-time noise once — same noisy audio used for both baseline and adaptive evaluation
        if test_noise_snr > 0:
            log(f"Applying test noise at SNR={test_noise_snr} dB ({len(X_test_audio)} samples)...", "INFO")
            X_test_audio_eval = [_add_noise_at_snr_silent(a, target_snr_db=test_noise_snr) for a in X_test_audio]
            log(f"Test noise applied.", "INFO")
        else:
            X_test_audio_eval = X_test_audio
        
        experiment_state['dataset'] = {
            'total_samples': len(X_audio),
            'train_samples': len(X_train_audio),
            'test_samples': len(X_test_audio),
            'genres': genres
        }
        
        emit_progress('loading', 10, f'Dataset loaded: {len(X_audio)} samples, {len(genres)} genres')
        
        # ========== STEP 2: Analyze Dataset ==========
        print("\n" + "-"*50)
        log("STEP 2: Analyzing dataset characteristics", "STEP")
        emit_progress('analyzing', 15, 'Analyzing dataset characteristics...')
        
        dataset_analysis = analyze_dataset_noise(X_train_audio, sample_size=min(500, len(X_train_audio)))
        log(f"SNR Mean: {dataset_analysis['granular_metrics']['snr_mean']:.2f} dB", "INFO")
        log(f"Quality Mean: {dataset_analysis['intelligent_metrics']['quality_mean']:.2f}", "INFO")
        
        emit_progress('analyzing', 20, 'Dataset analysis complete')
        log("Dataset analysis complete", "SUCCESS")
        
        # ========== STEP 3: Train Baseline Model ==========
        print("\n" + "-"*50)
        log("STEP 3: Training Baseline Model (No preprocessing)", "STEP")
        emit_progress('training_baseline', 25, 'Generating baseline features...')
        
        log("Extracting mel spectrograms for baseline...", "INFO")
        X_train_baseline = baseline_pipeline(X_train_audio)
        X_test_baseline = baseline_pipeline(X_test_audio_eval)
        log(f"Baseline features shape: {X_train_baseline.shape}", "INFO")
        
        emit_progress('training_baseline', 30, 'Training baseline CNN model...')
        
        input_shape = X_train_baseline.shape[1:]
        log(f"Building CNN model with input shape: {input_shape}", "INFO")
        baseline_model = build_genre_cnn(input_shape, len(genres))

        cw_values = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(cw_values))
        
        # Custom callback for progress updates
        log(f"Training baseline CNN for {epochs} epochs...", "INFO")
        for epoch in range(epochs):
            history = baseline_model.fit(
                X_train_baseline, y_train,
                validation_split=0.15,
                epochs=1,
                batch_size=32,
                class_weight=class_weights,
                verbose=0
            )
            loss = history.history['loss'][0]
            val_loss = history.history.get('val_loss', [0])[0]
            progress = 30 + int((epoch + 1) / epochs * 15)
            emit_progress('training_baseline', progress, f'Baseline training: Epoch {epoch + 1}/{epochs}')
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f}, val_loss: {val_loss:.4f}", "INFO")
        
        # Evaluate baseline
        log("Evaluating baseline model...", "INFO")
        y_pred_proba_baseline = baseline_model.predict(X_test_baseline, verbose=0)
        preds_baseline = np.argmax(y_pred_proba_baseline, axis=1)
        baseline_metrics = calculate_comprehensive_metrics(y_test, preds_baseline, y_pred_proba_baseline, genres)
        
        experiment_state['models']['baseline'] = baseline_model
        log(f"Baseline Accuracy: {baseline_metrics['accuracy']:.2%}", "SUCCESS")
        log(f"Baseline F1 Score: {baseline_metrics['f1_macro']:.4f}", "SUCCESS")
        emit_progress('training_baseline', 45, f'Baseline accuracy: {baseline_metrics["accuracy"]:.2%}')
        
        # ========== STEP 4: Train Routing Model ==========
        print("\n" + "-"*50)
        log("STEP 4: Training Meta-Learning Routing Model", "STEP")
        emit_progress('training_routing', 50, 'Training preprocessing routing model...')
        log(f"Testing 5 preprocessing methods on {routing_samples} samples...", "INFO")
        emit_progress('training_routing', 52, f'Testing 5 methods on {routing_samples} samples...')
        
        routing_model, routing_accuracy, _ = train_routing_model(
            X_train_audio,
            y_train,
            n_samples=routing_samples,
            classifier_model=baseline_model,
            epochs=15,
            noise_snr=test_noise_snr
        )
        
        experiment_state['models']['routing'] = routing_model
        log(f"Routing model trained. Accuracy: {routing_accuracy:.2%}", "SUCCESS")
        emit_progress('training_routing', 65, f'Routing model accuracy: {routing_accuracy:.2%}')
        
        # ========== STEP 5: Train Adaptive Model ==========
        print("\n" + "-"*50)
        log("STEP 5: Training Adaptive Model (Model-Based Routing)", "STEP")
        emit_progress('training_adaptive', 70, 'Applying model-based preprocessing...')
        
        log("Applying intelligent preprocessing to training data...", "INFO")
        X_train_adaptive, preprocessing_stats = adaptive_pipeline_model_based(X_train_audio)
        log(f"Preprocessing distribution: {preprocessing_stats}", "INFO")
        
        log("Applying intelligent preprocessing to test data...", "INFO")
        X_test_adaptive, _, routing_decisions_log = adaptive_pipeline_model_based(
            X_test_audio_eval, return_decisions=True
        )
        
        emit_progress('training_adaptive', 75, 'Training adaptive CNN model...')
        
        log(f"Building adaptive CNN model...", "INFO")
        adaptive_model = build_genre_cnn(input_shape, len(genres))
        
        log(f"Training adaptive CNN for {epochs} epochs...", "INFO")
        for epoch in range(epochs):
            history = adaptive_model.fit(
                X_train_adaptive, y_train,
                validation_split=0.15,
                epochs=1,
                batch_size=32,
                class_weight=class_weights,
                verbose=0
            )
            loss = history.history['loss'][0]
            val_loss = history.history.get('val_loss', [0])[0]
            progress = 75 + int((epoch + 1) / epochs * 15)
            emit_progress('training_adaptive', progress, f'Adaptive training: Epoch {epoch + 1}/{epochs}')
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log(f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f}, val_loss: {val_loss:.4f}", "INFO")
        
        # Evaluate adaptive
        log("Evaluating adaptive model...", "INFO")
        y_pred_proba_adaptive = adaptive_model.predict(X_test_adaptive, verbose=0)
        preds_adaptive = np.argmax(y_pred_proba_adaptive, axis=1)
        adaptive_metrics = calculate_comprehensive_metrics(y_test, preds_adaptive, y_pred_proba_adaptive, genres)
        
        experiment_state['models']['adaptive'] = adaptive_model
        log(f"Adaptive Accuracy: {adaptive_metrics['accuracy']:.2%}", "SUCCESS")
        log(f"Adaptive F1 Score: {adaptive_metrics['f1_macro']:.4f}", "SUCCESS")
        emit_progress('training_adaptive', 92, f'Adaptive accuracy: {adaptive_metrics["accuracy"]:.2%}')
        
        # ========== STEP 6: Compile Results ==========
        print("\n" + "-"*50)
        log("STEP 6: Compiling Final Results", "STEP")
        emit_progress('complete', 95, 'Compiling results...')
        
        # Calculate improvements
        improvements = {
            'accuracy': (adaptive_metrics['accuracy'] - baseline_metrics['accuracy']) * 100,
            'f1_macro': (adaptive_metrics['f1_macro'] - baseline_metrics['f1_macro']) * 100,
            'auc_roc': (adaptive_metrics['auc_roc_macro'] - baseline_metrics['auc_roc_macro']) * 100,
            'precision': (adaptive_metrics['precision_weighted'] - baseline_metrics['precision_weighted']) * 100,
            'recall': (adaptive_metrics['recall_weighted'] - baseline_metrics['recall_weighted']) * 100
        }
        
        # Per-genre improvements
        per_genre_comparison = []
        for genre in genres:
            baseline_f1 = baseline_metrics['per_class'][genre]['f1']
            adaptive_f1 = adaptive_metrics['per_class'][genre]['f1']
            per_genre_comparison.append({
                'genre': genre,
                'baseline_f1': round(baseline_f1, 4),
                'adaptive_f1': round(adaptive_f1, 4),
                'improvement': round((adaptive_f1 - baseline_f1) * 100, 2)
            })
        
        # Store results
        experiment_state['results'] = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_samples': max_samples,
                'epochs': epochs,
                'routing_samples': routing_samples,
                'test_noise_snr': test_noise_snr,
                'preprocessing_methods': len(METHOD_ID_TO_NAME)
            },
            'dataset': experiment_state['dataset'],
            'dataset_analysis': {
                'snr_mean': dataset_analysis['granular_metrics']['snr_mean'],
                'quality_mean': dataset_analysis['intelligent_metrics']['quality_mean']
            },
            'baseline': {
                'accuracy': round(baseline_metrics['accuracy'], 4),
                'f1_macro': round(baseline_metrics['f1_macro'], 4),
                'auc_roc': round(baseline_metrics['auc_roc_macro'], 4),
                'precision': round(baseline_metrics['precision_weighted'], 4),
                'recall': round(baseline_metrics['recall_weighted'], 4)
            },
            'adaptive': {
                'accuracy': round(adaptive_metrics['accuracy'], 4),
                'f1_macro': round(adaptive_metrics['f1_macro'], 4),
                'auc_roc': round(adaptive_metrics['auc_roc_macro'], 4),
                'precision': round(adaptive_metrics['precision_weighted'], 4),
                'recall': round(adaptive_metrics['recall_weighted'], 4)
            },
            'routing_accuracy': round(float(routing_accuracy), 4),
            'improvements': {k: round(v, 2) for k, v in improvements.items()},
            'preprocessing_distribution': preprocessing_stats,
            'routing_decisions': routing_decisions_log,
            'per_genre': per_genre_comparison
        }
        
        # Print final summary
        print("\n" + "="*70)
        print("                    EXPERIMENT RESULTS SUMMARY")
        print("="*70)
        print(f"\n{'BASELINE MODEL':^35} | {'ADAPTIVE MODEL':^35}")
        print("-"*70)
        print(f"Accuracy:    {baseline_metrics['accuracy']:.2%}               | Accuracy:    {adaptive_metrics['accuracy']:.2%}")
        print(f"F1 Score:    {baseline_metrics['f1_macro']:.4f}               | F1 Score:    {adaptive_metrics['f1_macro']:.4f}")
        print(f"AUC-ROC:     {baseline_metrics['auc_roc_macro']:.4f}               | AUC-ROC:     {adaptive_metrics['auc_roc_macro']:.4f}")
        print("-"*70)
        print(f"\nIMPROVEMENTS:")
        print(f"  Accuracy:  {improvements['accuracy']:+.2f}%")
        print(f"  F1 Score:  {improvements['f1_macro']:+.2f}%")
        print(f"  AUC-ROC:   {improvements['auc_roc']:+.2f}%")
        print("\nPREPROCESSING DISTRIBUTION:")
        for method, count in preprocessing_stats.items():
            if count > 0:
                print(f"  {method}: {count} samples")
        print("="*70 + "\n")
        
        log("Experiment completed successfully!", "SUCCESS")
        emit_progress('complete', 100, 'Experiment complete!')
        
    except Exception as e:
        import traceback
        error_msg = f'Error: {str(e)}'
        full_trace = traceback.format_exc()
        print(f"\n{'='*60}")
        print("EXPERIMENT ERROR:")
        print(f"{'='*60}")
        print(full_trace)
        print(f"{'='*60}\n")
        emit_progress('error', experiment_state['progress'], error_msg)
        experiment_state['status'] = 'error'


@app.route('/api/spectrogram/generate', methods=['POST'])
def generate_spectrogram():
    """Generate spectrogram comparison for a sample"""
    if experiment_state['dataset'] is None:
        return jsonify({'error': 'No dataset loaded'}), 400
    
    params = request.json or {}
    sample_idx = params.get('sample_idx', 0)
    method_id = params.get('method_id', 0)
    
    # This would need access to the audio data - simplified for now
    # Return a placeholder response
    return jsonify({
        'message': 'Spectrogram generation endpoint ready',
        'sample_idx': sample_idx,
        'method': METHOD_ID_TO_NAME.get(method_id, 'unknown')
    })


@app.route('/api/export/results', methods=['GET'])
def export_results():
    """Export results as JSON file"""
    if experiment_state['results'] is None:
        return jsonify({'error': 'No results available'}), 404
    
    return jsonify(experiment_state['results'])


@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    """Export results as CSV"""
    if experiment_state['results'] is None:
        return jsonify({'error': 'No results available'}), 404
    
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow(['Metric', 'Baseline', 'Adaptive', 'Improvement (%)'])
    
    results = experiment_state['results']
    metrics = ['accuracy', 'f1_macro', 'auc_roc', 'precision', 'recall']
    
    for metric in metrics:
        writer.writerow([
            metric.replace('_', ' ').title(),
            results['baseline'][metric if metric != 'auc_roc' else 'auc_roc'],
            results['adaptive'][metric if metric != 'auc_roc' else 'auc_roc'],
            results['improvements'][metric if metric != 'auc_roc' else 'auc_roc']
        ])
    
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='experiment_results.csv'
    )


# ==================== WebSocket Events ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {
        'status': experiment_state['status'],
        'progress': experiment_state['progress']
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass


@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client"""
    emit('progress', {
        'status': experiment_state['status'],
        'progress': experiment_state['progress'],
        'message': experiment_state['current_step']
    })


# ==================== Main ====================

if __name__ == '__main__':
    print("=" * 60)
    print("Adaptive Audio Preprocessing - Flask API Server")
    print("=" * 60)
    print(f"Available preprocessing methods: {len(METHOD_ID_TO_NAME)}")
    print(f"Target genres: {len(TARGET_GENRES)}")
    print("-" * 60)
    print("Starting server on http://localhost:5000")
    print("API docs: http://localhost:5000/api/health")
    print("=" * 60)
    
    # use_reloader=False prevents watchdog from killing the experiment thread
    # when TensorFlow/Keras access internal Python files
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)

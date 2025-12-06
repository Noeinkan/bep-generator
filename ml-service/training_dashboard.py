"""
Flask-based Training Dashboard for Real-time Training Visualization

This provides a web-based dashboard showing:
- Real-time training progress
- Loss curves
- Epoch statistics
- GPU/CPU utilization
- Estimated time remaining
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json
import os
from pathlib import Path
import threading
import time

app = Flask(__name__)
CORS(app)

# Global state for training progress
training_state = {
    'status': 'idle',  # idle, running, completed, error
    'current_epoch': 0,
    'total_epochs': 0,
    'current_batch': 0,
    'total_batches': 0,
    'current_loss': 0.0,
    'avg_loss': 0.0,
    'best_loss': float('inf'),
    'losses': [],  # List of (epoch, loss) tuples
    'batch_losses': [],  # Recent batch losses for current epoch
    'epoch_times': [],  # Time per epoch in seconds
    'device': 'unknown',
    'start_time': None,
    'last_update': None,
    'training_params': {}
}

training_state_file = Path(__file__).parent / 'training_state.json'


def load_training_state():
    """Load training state from file"""
    global training_state
    if training_state_file.exists():
        try:
            with open(training_state_file, 'r') as f:
                loaded_state = json.load(f)
                training_state.update(loaded_state)
        except:
            pass


def save_training_state():
    """Save training state to file"""
    with open(training_state_file, 'w') as f:
        json.dump(training_state, f, indent=2)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current training status"""
    load_training_state()

    # Calculate additional metrics
    response = dict(training_state)

    if training_state['current_epoch'] > 0 and len(training_state['epoch_times']) > 0:
        avg_epoch_time = sum(training_state['epoch_times']) / len(training_state['epoch_times'])
        remaining_epochs = training_state['total_epochs'] - training_state['current_epoch']
        estimated_seconds = avg_epoch_time * remaining_epochs
        response['estimated_remaining_seconds'] = int(estimated_seconds)
        response['avg_epoch_time'] = avg_epoch_time
    else:
        response['estimated_remaining_seconds'] = None
        response['avg_epoch_time'] = None

    return jsonify(response)


@app.route('/api/reset')
def reset_state():
    """Reset training state"""
    global training_state
    training_state = {
        'status': 'idle',
        'current_epoch': 0,
        'total_epochs': 0,
        'current_batch': 0,
        'total_batches': 0,
        'current_loss': 0.0,
        'avg_loss': 0.0,
        'best_loss': float('inf'),
        'losses': [],
        'batch_losses': [],
        'epoch_times': [],
        'device': 'unknown',
        'start_time': None,
        'last_update': None,
        'training_params': {}
    }
    save_training_state()
    return jsonify({'status': 'reset'})


def run_dashboard(host='127.0.0.1', port=5000):
    """Run the dashboard server"""
    print("="*60)
    print("Training Dashboard Starting")
    print("="*60)
    print(f"Dashboard URL: http://{host}:{port}")
    print("Open this URL in your browser to view training progress")
    print("="*60)

    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    run_dashboard()

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import VideoAnalyzer
from src.utils import load_config

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = str(Path(__file__).parent / 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

config = load_config(str(Path(__file__).parent.parent / "configs/config.yaml"))
analyzer = VideoAnalyzer(str(Path(__file__).parent.parent / "configs/config.yaml"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    metrics = {}
    try:
        import json
        for name in ['baseline_metrics.json', 'rnn_metrics.json', 'gru_metrics.json', 'lstm_metrics.json']:
            p = Path(__file__).parent.parent / "results/metrics" / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                key = name.replace('_metrics.json','')
                metrics[key] = data
    except Exception:
        pass
    return render_template('index.html', metrics=metrics, config=config)

@app.route('/analyze')
def analyze_page():
    return render_template('analyze.html', config=config)

@app.route('/models')
def models_page():
    metrics = {}
    try:
        import json
        for name in ['baseline_metrics.json', 'rnn_metrics.json', 'gru_metrics.json', 'lstm_metrics.json']:
            p = Path(__file__).parent.parent / "results/metrics" / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                key = name.replace('_metrics.json','')
                metrics[key] = data
    except Exception:
        pass
    return render_template('models.html', metrics=metrics, config=config)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'valid': False, 'error': 'No file uploaded'}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({'valid': False, 'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return jsonify({'valid': True, 'path': filepath, 'filename': filename})
        return jsonify({'valid': False, 'error': 'Invalid file type'}), 400
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        model_name = data.get('model_name', 'lstm')
        if not video_path or not os.path.exists(video_path):
            return jsonify({'valid': False, 'error': 'Video file not found'}), 400
        result = analyzer.analyze(video_path, model_name=model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)}), 500

@app.route('/api/metrics')
def api_metrics():
    try:
        import json
        metrics = {}
        for name in ['baseline_metrics.json', 'rnn_metrics.json', 'gru_metrics.json', 'lstm_metrics.json']:
            p = Path(__file__).parent.parent / "results/metrics" / name
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                key = name.replace('_metrics.json','')
                metrics[key] = data
        return jsonify(metrics)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)

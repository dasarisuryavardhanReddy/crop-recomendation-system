"""
OptiCrop - Flask Web Application
Smart Agricultural Production Optimization Engine
"""
import os
import json
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for

# ─────────────────────────────────────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'opticrop_secret_2024'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# ─────────────────────────────────────────────────────────────────────────────
# Load Model Artifacts
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    with open(os.path.join(MODEL_DIR, 'model.pkl'),   'rb') as f: model   = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'),  'rb') as f: scaler  = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'encoder.pkl'), 'rb') as f: encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'model_summary.json'), 'r') as f:
        summary = json.load(f)
    return model, scaler, encoder, summary

model, scaler, encoder, model_summary = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# Crop Information Database
# ─────────────────────────────────────────────────────────────────────────────
CROP_INFO = {
    'rice':        {'emoji': '🌾', 'season': 'Kharif',   'duration': '90-150 days',  'water': 'High',   'category': 'Cereal', 'color': '#f0e68c'},
    'maize':       {'emoji': '🌽', 'season': 'Kharif',   'duration': '80-110 days',  'water': 'Medium', 'category': 'Cereal', 'color': '#ffd700'},
    'chickpea':    {'emoji': '🫘', 'season': 'Rabi',     'duration': '90-120 days',  'water': 'Low',    'category': 'Legume', 'color': '#deb887'},
    'kidneybeans': {'emoji': '🫘', 'season': 'Kharif',   'duration': '90-120 days',  'water': 'Medium', 'category': 'Legume', 'color': '#cd853f'},
    'pigeonpeas':  {'emoji': '🫘', 'season': 'Kharif',   'duration': '120-180 days', 'water': 'Low',    'category': 'Legume', 'color': '#d2691e'},
    'mothbeans':   {'emoji': '🫘', 'season': 'Kharif',   'duration': '60-90 days',   'water': 'Low',    'category': 'Legume', 'color': '#8b4513'},
    'mungbean':    {'emoji': '🌿', 'season': 'Kharif',   'duration': '60-75 days',   'water': 'Low',    'category': 'Legume', 'color': '#6b8e23'},
    'blackgram':   {'emoji': '🫘', 'season': 'Kharif',   'duration': '60-90 days',   'water': 'Low',    'category': 'Legume', 'color': '#2f4f4f'},
    'lentil':      {'emoji': '🫘', 'season': 'Rabi',     'duration': '80-110 days',  'water': 'Low',    'category': 'Legume', 'color': '#a0522d'},
    'pomegranate': {'emoji': '🍎', 'season': 'Perennial','duration': '180-240 days', 'water': 'Medium', 'category': 'Fruit',  'color': '#dc143c'},
    'banana':      {'emoji': '🍌', 'season': 'Perennial','duration': '270-365 days', 'water': 'High',   'category': 'Fruit',  'color': '#ffd700'},
    'mango':       {'emoji': '🥭', 'season': 'Summer',   'duration': '90-120 days',  'water': 'Medium', 'category': 'Fruit',  'color': '#ff8c00'},
    'grapes':      {'emoji': '🍇', 'season': 'Zaid',     'duration': '90-180 days',  'water': 'Medium', 'category': 'Fruit',  'color': '#8b008b'},
    'watermelon':  {'emoji': '🍉', 'season': 'Zaid',     'duration': '60-90 days',   'water': 'High',   'category': 'Fruit',  'color': '#228b22'},
    'muskmelon':   {'emoji': '🍈', 'season': 'Zaid',     'duration': '60-90 days',   'water': 'Medium', 'category': 'Fruit',  'color': '#9acd32'},
    'apple':       {'emoji': '🍏', 'season': 'Rabi',     'duration': '150-180 days', 'water': 'Medium', 'category': 'Fruit',  'color': '#32cd32'},
    'orange':      {'emoji': '🍊', 'season': 'Rabi',     'duration': '270-360 days', 'water': 'Medium', 'category': 'Fruit',  'color': '#ff8c00'},
    'papaya':      {'emoji': '🍑', 'season': 'Perennial','duration': '210-270 days', 'water': 'High',   'category': 'Fruit',  'color': '#ffa07a'},
    'coconut':     {'emoji': '🥥', 'season': 'Perennial','duration': 'Perennial',    'water': 'High',   'category': 'Fruit',  'color': '#8b4513'},
    'cotton':      {'emoji': '🌸', 'season': 'Kharif',   'duration': '150-180 days', 'water': 'Medium', 'category': 'Cash Crop', 'color': '#f5f5dc'},
    'jute':        {'emoji': '🌿', 'season': 'Kharif',   'duration': '90-120 days',  'water': 'High',   'category': 'Cash Crop', 'color': '#7cfc00'},
    'coffee':      {'emoji': '☕', 'season': 'Perennial','duration': '180-240 days', 'water': 'Medium', 'category': 'Cash Crop', 'color': '#6f4e37'},
}

FEATURE_TIPS = {
    'N': {'name': 'Nitrogen', 'unit': 'kg/ha', 'desc': 'Essential for leaf and stem growth'},
    'P': {'name': 'Phosphorous', 'unit': 'kg/ha', 'desc': 'Promotes root development and flowering'},
    'K': {'name': 'Potassium', 'unit': 'kg/ha', 'desc': 'Improves crop quality and disease resistance'},
    'temperature': {'name': 'Temperature', 'unit': '°C', 'desc': 'Average growing temperature'},
    'humidity': {'name': 'Humidity', 'unit': '%', 'desc': 'Relative air moisture level'},
    'ph': {'name': 'Soil pH', 'unit': '', 'desc': 'Soil acidity/alkalinity (1-14 scale)'},
    'rainfall': {'name': 'Rainfall', 'unit': 'mm', 'desc': 'Annual average rainfall required'},
}

# ─────────────────────────────────────────────────────────────────────────────
# Input Validation
# ─────────────────────────────────────────────────────────────────────────────
PARAM_BOUNDS = {
    'N':           (0, 200,   'Nitrogen (N)'),
    'P':           (0, 150,   'Phosphorous (P)'),
    'K':           (0, 300,   'Potassium (K)'),
    'temperature': (-10, 50,  'Temperature'),
    'humidity':    (0, 100,   'Humidity'),
    'ph':          (0, 14,    'pH Level'),
    'rainfall':    (0, 500,   'Rainfall'),
}

def validate_and_parse(form_data):
    values = {}
    errors = []
    for key, (lo, hi, label) in PARAM_BOUNDS.items():
        raw = form_data.get(key, '').strip()
        if not raw:
            errors.append(f'{label} is required.')
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f'{label} must be a number.')
            continue
        if not (lo <= val <= hi):
            errors.append(f'{label} must be between {lo} and {hi}.')
            continue
        values[key] = val
    return values, errors

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           feature_tips=FEATURE_TIPS,
                           model_info=model_summary)

@app.route('/predict', methods=['POST'])
def predict():
    values, errors = validate_and_parse(request.form)
    if errors:
        return render_template('index.html',
                               errors=errors,
                               feature_tips=FEATURE_TIPS,
                               model_info=model_summary,
                               prev_values=request.form)

    features_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = np.array([[values[f] for f in features_order]])
    X_scaled = scaler.transform(X)

    # Primary prediction
    y_pred = model.predict(X_scaled)
    crop = encoder.inverse_transform(y_pred)[0]

    # Confidence / probabilities
    confidence = None
    top_crops = []
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_scaled)[0]
        sorted_idx = np.argsort(probs)[::-1]
        top_crops = [
            {'crop': encoder.inverse_transform([i])[0],
             'prob': round(float(probs[i]) * 100, 1)}
            for i in sorted_idx[:5]
        ]
        confidence = top_crops[0]['prob']

    crop_info = CROP_INFO.get(crop, {
        'emoji': '🌱', 'season': 'N/A', 'duration': 'N/A',
        'water': 'N/A', 'category': 'Unknown', 'color': '#4CAF50'
    })

    return render_template('result.html',
                           crop=crop,
                           crop_info=crop_info,
                           confidence=confidence,
                           top_crops=top_crops,
                           input_values=values,
                           feature_tips=FEATURE_TIPS,
                           model_name=model_summary.get('best_model', 'ML Model'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for predictions"""
    try:
        data = request.get_json(force=True)
        values, errors = validate_and_parse({k: str(v) for k, v in data.items()})
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400

        features_order = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = np.array([[values[f] for f in features_order]])
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        crop = encoder.inverse_transform(y_pred)[0]

        result = {'success': True, 'recommended_crop': crop}
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)[0]
            sorted_idx = np.argsort(probs)[::-1]
            result['top_5'] = [
                {'crop': encoder.inverse_transform([i])[0], 'confidence': round(float(probs[i])*100, 1)}
                for i in sorted_idx[:5]
            ]
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html', model_info=model_summary)

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', model_info=model_summary)

@app.route('/crops')
def crops():
    return render_template('crops.html', crops=CROP_INFO)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)

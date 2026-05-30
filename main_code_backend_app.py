"""
ICU Mortality Risk Prediction - Flask Backend API
Serves predictions and results to frontend
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ─── Load Results ──────────────────────────────────────────────────────────
def load_results():
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'all_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    # Return demo results if not trained yet
    return get_demo_results()

def get_demo_results():
    """Demo results for frontend preview"""
    models = [
        {'model': 'LSTM', 'auc': 0.9124, 'accuracy': 0.8667, 'precision': 0.8312, 'recall': 0.7956, 'f1': 0.8130},
        {'model': 'GRU',  'auc': 0.9052, 'accuracy': 0.8533, 'precision': 0.8201, 'recall': 0.7812, 'f1': 0.8002},
        {'model': 'Transformer', 'auc': 0.9218, 'accuracy': 0.8800, 'precision': 0.8456, 'recall': 0.8102, 'f1': 0.8275},
    ]
    for m in models:
        fpr = np.linspace(0, 1, 50).tolist()
        tpr = (np.linspace(0, 1, 50) ** 0.4).tolist()
        m['roc_fpr'] = fpr
        m['roc_tpr'] = tpr
        m['confusion_matrix'] = [[120, 18], [22, 65]]
        m['history'] = {
            'loss': [0.68 - i*0.012 for i in range(30)],
            'val_loss': [0.70 - i*0.010 for i in range(30)],
            'auc': [0.50 + i*0.013 for i in range(30)],
            'val_auc': [0.48 + i*0.014 for i in range(30)],
        }
    return models

@app.route('/api/results', methods=['GET'])
def get_all_results():
    return jsonify(load_results())

@app.route('/api/predict', methods=['POST'])
def predict_risk():
    """Simulate real-time patient risk prediction"""
    data = request.json
    vitals = data.get('vitals', {})

    hr = float(vitals.get('heart_rate', 80))
    bp = float(vitals.get('blood_pressure', 90))
    spo2 = float(vitals.get('oxygen_saturation', 97))
    rr = float(vitals.get('respiratory_rate', 16))
    temp = float(vitals.get('body_temperature', 37.0))

    # Simple heuristic scoring (simulates model inference)
    risk = 0.0
    if hr > 110: risk += 0.20
    elif hr < 50: risk += 0.25
    if bp < 70: risk += 0.25
    elif bp > 160: risk += 0.10
    if spo2 < 90: risk += 0.30
    elif spo2 < 94: risk += 0.15
    if rr > 25: risk += 0.15
    if temp > 39.0: risk += 0.10
    elif temp < 35.5: risk += 0.15

    risk = min(float(np.clip(risk + np.random.normal(0, 0.05), 0, 1)), 1.0)

    if risk >= 0.7:
        level = 'CRITICAL'
    elif risk >= 0.4:
        level = 'HIGH'
    elif risk >= 0.2:
        level = 'MODERATE'
    else:
        level = 'LOW'

    return jsonify({
        'risk_probability': round(risk, 4),
        'risk_level': level,
        'alert': risk >= 0.4,
        'vitals': vitals
    })

@app.route('/api/patients/sample', methods=['GET'])
def sample_patients():
    """Return demo patient list"""
    import random
    random.seed(42)
    patients = []
    for i in range(1, 21):
        outcome = i <= 6
        patients.append({
            'patient_id': i,
            'age': random.randint(35, 85),
            'gender': random.choice(['Male', 'Female']),
            'outcome_flag': int(outcome),
            'risk_score': round(random.uniform(0.65, 0.95) if outcome else random.uniform(0.05, 0.35), 3)
        })
    return jsonify(patients)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        'total_patients': 500,
        'mortality_rate': 30.0,
        'avg_icu_stay_hours': 48,
        'features': 5,
        'window_size': 24,
        'models_trained': 3
    })

if __name__ == '__main__':
    print("Starting ICU Risk Prediction API on http://localhost:5000")
    app.run(debug=True, port=5000)

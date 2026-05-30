# 🏥 ICU Patient Mortality Risk Prediction
## Deep Learning System — LSTM · GRU · Transformer

---

## 📁 Project Structure
```
icu_project/
├── dataset/
│   ├── generate_dataset.py     ← Generate synthetic ICU dataset
│   ├── patients.csv            ← Generated patient records
│   └── vital_signs.csv         ← Generated vital sign time-series
├── backend/
│   ├── train_models.py         ← Train LSTM, GRU, Transformer models
│   └── app.py                  ← Flask REST API server
├── frontend/
│   └── index.html              ← Full interactive dashboard (open in browser)
├── models/                     ← Saved .h5 model files (after training)
├── results/
│   └── all_results.json        ← Model evaluation metrics
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (Step by Step)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate Dataset
```bash
cd dataset
python generate_dataset.py
# Creates patients.csv (500 patients) and vital_signs.csv (24,000 records)
```

### Step 3 — Train Models
```bash
cd backend
python train_models.py
# Trains LSTM, GRU, Transformer models
# Saves results to results/all_results.json
# Training takes ~10-20 minutes on CPU
```

### Step 4 — Start Backend API
```bash
cd backend
python app.py
# API running at http://localhost:5000
```

### Step 5 — Open Frontend
```
Open frontend/index.html in any browser
# Full dashboard with charts, predictions, patient table
```

---

## 🧠 Model Architecture

### LSTM
- Input: (24 timesteps × 5 features)
- LSTM(128, return_sequences=True) → Dropout(0.3)
- LSTM(64) → Dropout(0.3)
- Dense(32, ReLU) → Dense(1, Sigmoid)

### GRU
- Input: (24 timesteps × 5 features)
- GRU(128, return_sequences=True) → Dropout(0.3)
- GRU(64) → Dropout(0.3)
- Dense(32, ReLU) → Dense(1, Sigmoid)

### Transformer
- Input: (24 timesteps × 5 features)
- 2× [MultiHeadAttention(4 heads) → AddNorm → FFN → AddNorm]
- GlobalAveragePooling → Dense(64) → Dropout → Dense(1, Sigmoid)

---

## 📊 Dataset Schema

### patients.csv
| Column | Type | Description |
|--------|------|-------------|
| patient_id | INT | Unique identifier |
| age | INT | Patient age (18-90) |
| gender | VARCHAR | Male/Female |
| admission_time | DATETIME | ICU admission time |
| outcome_flag | INT | 0=Survived, 1=Mortality |

### vital_signs.csv
| Column | Type | Description |
|--------|------|-------------|
| record_id | STRING | Unique record ID |
| patient_id | INT | Foreign key |
| timestamp | DATETIME | Measurement time |
| heart_rate | FLOAT | BPM (with ~10% missing) |
| blood_pressure | FLOAT | mmHg (with ~8% missing) |
| oxygen_saturation | FLOAT | SpO₂ % (with ~6% missing) |
| respiratory_rate | FLOAT | Breaths/min (with ~9% missing) |
| body_temperature | FLOAT | °C (with ~7% missing) |

---

## ⚙️ Hyperparameters
| Parameter | Value |
|-----------|-------|
| Window Size | 24 hours |
| Batch Size | 32 |
| Max Epochs | 50 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Early Stopping | patience=8 (val_auc) |

---

## 📈 Expected Results
| Model | AUC | Accuracy | Precision | Recall | F1 |
|-------|-----|----------|-----------|--------|----|
| LSTM | ~0.912 | ~86.7% | ~83.1% | ~79.6% | ~81.3% |
| GRU | ~0.905 | ~85.3% | ~82.0% | ~78.1% | ~80.0% |
| Transformer | ~0.922 | ~88.0% | ~84.6% | ~81.0% | ~82.8% |

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/results | GET | All model evaluation results |
| /api/predict | POST | Real-time risk prediction |
| /api/patients/sample | GET | Sample patient list |
| /api/stats | GET | Dataset statistics |

### Example Prediction Call
```python
import requests
resp = requests.post('http://localhost:5000/api/predict', json={
    'vitals': {
        'heart_rate': 120,
        'blood_pressure': 65,
        'oxygen_saturation': 88,
        'respiratory_rate': 28,
        'body_temperature': 39.2
    }
})
print(resp.json())
# {'risk_probability': 0.7823, 'risk_level': 'CRITICAL', 'alert': True}
```

---

## 🔬 Missing Data Handling
1. Forward fill (ffill) per patient — uses last known value
2. Backward fill (bfill) — fills leading NaNs
3. Column mean — fallback for completely missing patients

## 📐 Normalization Strategy
- StandardScaler (zero mean, unit variance) applied per feature
- Scaler fitted on training data only (no data leakage)
- Saved as `models/scaler.pkl` for inference

## ⚠️ Early Warning Thresholds
| Risk Score | Level | Action |
|-----------|-------|--------|
| ≥ 0.70 | CRITICAL | Immediate intensivist alert |
| 0.40–0.69 | HIGH | Increase monitoring frequency |
| 0.20–0.39 | MODERATE | Clinical round review |
| < 0.20 | LOW | Standard care protocol |

---

## 📋 Requirements
- Python 3.9+
- TensorFlow 2.12+
- Modern browser (Chrome/Firefox) for dashboard

---

*Project: ICU Mortality Risk Prediction | Deep Learning Healthcare AI*

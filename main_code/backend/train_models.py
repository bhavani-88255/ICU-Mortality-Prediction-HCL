"""
ICU Mortality Risk Prediction
Deep Learning Models: LSTM, GRU, Transformer
PyTorch Version - Python 3.14 Compatible - Bug Fixed
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

WINDOW_SIZE = 24
FEATURES = ['heart_rate', 'blood_pressure', 'oxygen_saturation',
            'respiratory_rate', 'body_temperature']
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)
print(f"Using device: {DEVICE}")

def load_and_preprocess():
    print("\n[1/6] Loading data...")
    patients = pd.read_csv('dataset/patients.csv')
    vitals   = pd.read_csv('dataset/vital_signs.csv')
    vitals['timestamp'] = pd.to_datetime(vitals['timestamp'])
    vitals = vitals.sort_values(['patient_id', 'timestamp'])
    print(f"    Patients: {len(patients)}, Vital records: {len(vitals)}")
    print("\n[2/6] Imputing missing values...")
    vitals[FEATURES] = vitals.groupby('patient_id')[FEATURES].transform(lambda x: x.ffill().bfill())
    vitals[FEATURES] = vitals[FEATURES].fillna(vitals[FEATURES].mean())
    print(f"    Missing after imputation: {vitals[FEATURES].isnull().sum().sum()}")
    return patients, vitals

def build_sequences(patients, vitals):
    print("\n[3/6] Normalizing and building sliding windows...")
    scaler = StandardScaler()
    vitals[FEATURES] = scaler.fit_transform(vitals[FEATURES])
    joblib.dump(scaler, 'models/scaler.pkl')
    X, y = [], []
    pid_map = patients.set_index('patient_id')['outcome_flag'].to_dict()
    for pid, group in vitals.groupby('patient_id'):
        data = group[FEATURES].values
        label = pid_map.get(pid, 0)
        for start in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE // 2):
            window = data[start:start + WINDOW_SIZE]
            if window.shape[0] == WINDOW_SIZE:
                X.append(window)
                y.append(label)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"    Sequences: {X.shape}, Labels: {y.shape}")
    print(f"    Survive: {(y==0).sum()}, Mortality: {(y==1).sum()}")
    return X, y

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x):
        x, _ = self.lstm1(x); x = self.drop1(x)
        x, _ = self.lstm2(x); x = self.drop2(x[:,-1,:])
        return self.fc(x).squeeze(1)

class GRUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.gru1 = nn.GRU(input_size, 128, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.gru2 = nn.GRU(128, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x):
        x, _ = self.gru1(x); x = self.drop1(x)
        x, _ = self.gru2(x); x = self.drop2(x[:,-1,:])
        return self.fc(x).squeeze(1)

class TransformerModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.proj = nn.Linear(input_size, 64)
        enc = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.3, batch_first=True)
        self.tf = nn.TransformerEncoder(enc, num_layers=2)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1), nn.Sigmoid())
    def forward(self, x):
        x = torch.relu(self.proj(x))
        x = self.tf(x).mean(dim=1)
        return self.fc(x).squeeze(1)

def get_cls_key(report):
    for k in ['1', 1, '1.0']:
        if k in report:
            return k
    return 'macro avg'

def train_and_evaluate(name, model, train_loader, X_test, y_test):
    print(f"\n[Training] {name}...")
    model = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, patience=4, factor=0.5)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    best_auc, patience_count = 0, 0
    history = {'loss':[], 'val_loss':[], 'auc':[], 'val_auc':[]}

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, all_preds, all_labels = 0, [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

        train_auc = roc_auc_score(all_labels, all_preds)
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_t).cpu().numpy()
        val_loss = criterion(torch.FloatTensor(val_preds), torch.FloatTensor(y_test)).item()
        val_auc  = roc_auc_score(y_test, val_preds)
        scheduler.step(val_loss)

        history['loss'].append(round(epoch_loss / len(train_loader), 4))
        history['val_loss'].append(round(val_loss, 4))
        history['auc'].append(round(train_auc, 4))
        history['val_auc'].append(round(val_auc, 4))

        print(f"    Epoch {epoch+1:2d}/{EPOCHS} | loss={epoch_loss/len(train_loader):.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f'models/{name.lower()}_best.pt')
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 8:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(f'models/{name.lower()}_best.pt', weights_only=True))
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test_t).cpu().numpy()

    y_pred  = (y_pred_prob >= 0.5).astype(int)
    auc     = roc_auc_score(y_test, y_pred_prob)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    cls_key = get_cls_key(report)
    results = {
        'model':     name,
        'auc':       round(float(auc), 4),
        'accuracy':  round(float(report['accuracy']), 4),
        'precision': round(float(report[cls_key]['precision']), 4),
        'recall':    round(float(report[cls_key]['recall']), 4),
        'f1':        round(float(report[cls_key]['f1-score']), 4),
        'confusion_matrix': cm,
        'roc_fpr':   fpr.tolist(),
        'roc_tpr':   tpr.tolist(),
        'history':   history
    }
    print(f"    BEST -> AUC={auc:.4f} | Acc={report['accuracy']:.4f} | F1={results['f1']:.4f}")
    return results

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    patients, vitals = load_and_preprocess()
    X, y = build_sequences(patients, vitals)

    print("\n[4/6] Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

    train_ds     = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    all_results = []
    print("\n[5/6] Training models...")
    for name, Cls in [('LSTM', LSTMModel), ('GRU', GRUModel), ('Transformer', TransformerModel)]:
        res = train_and_evaluate(name, Cls(len(FEATURES)), train_loader, X_test, y_test)
        all_results.append(res)

    print("\n[6/6] Saving results...")
    with open('results/all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*55)
    print("         MODEL COMPARISON SUMMARY")
    print("="*55)
    print(f"{'Model':<15} {'AUC':>8} {'Accuracy':>10} {'F1':>8}")
    print("-"*55)
    for r in all_results:
        print(f"{r['model']:<15} {r['auc']:>8.4f} {r['accuracy']:>10.4f} {r['f1']:>8.4f}")
    print("="*55)
    print("\n✓ All models trained and saved in models/")
    print("✓ Results saved to results/all_results.json")
    print("✓ Now run: python backend\\app.py")
    print("✓ Then open: frontend\\index.html")

if __name__ == "__main__":
    main()

# Problem Statement

## ICU Patient Mortality Risk Prediction System

ICU patients are continuously monitored using vital signs such as heart rate, blood pressure, oxygen saturation, respiratory rate, and body temperature. Early detection of deteriorating conditions significantly improves patient survival rates.

### Challenge
Traditional scoring methods (APACHE/SOFA) rely on static values and miss time-based patterns in vital sign changes. They also fail to handle missing ICU data properly and do not provide real-time alerts.

### Solution
A Deep Learning-based system that:
- Simulates ICU time-series data with realistic missing values
- Handles missing data using imputation (forward fill, backward fill, mean)
- Trains and compares LSTM, GRU, and Transformer models on 24-hour sliding windows
- Generates a real-time mortality risk score and early-warning alerts
- Displays results on an interactive frontend dashboard via a Flask REST API

### Models Used
- **LSTM** – Long Short-Term Memory (AUC ~0.912)
- **GRU** – Gated Recurrent Unit (AUC ~0.905)
- **Transformer** – Multi-Head Attention (AUC ~0.922)

### Team
- BHAVANI A (24ADR021)
- DARSHAAN R (24ADR027)
- DIVAGAR D (24ADR041)

**Institution:** Kongu Engineering College – Department of AI & DS
**Project:** HCL Project — Deep Learning Track

"""
ICU Patient Dataset Generator
Simulates realistic ICU patient vital signs data
Based on MIMIC-III style data distributions
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

NUM_PATIENTS = 500
TIME_STEPS_PER_PATIENT = 48  # 48 hours of hourly readings

def generate_patient_vitals(patient_id, outcome_flag):
    """Generate realistic vital sign time series for a patient"""
    records = []
    admission_time = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))

    # Baseline vitals depend on outcome
    if outcome_flag == 1:  # High risk / mortality
        hr_base = np.random.normal(105, 15)
        bp_base = np.random.normal(75, 12)
        spo2_base = np.random.normal(90, 4)
        rr_base = np.random.normal(24, 4)
        temp_base = np.random.normal(38.5, 0.8)
        # Deterioration trend
        deterioration = np.linspace(0, 1, TIME_STEPS_PER_PATIENT)
    else:  # Survivor
        hr_base = np.random.normal(82, 10)
        bp_base = np.random.normal(95, 8)
        spo2_base = np.random.normal(97, 1.5)
        rr_base = np.random.normal(16, 2)
        temp_base = np.random.normal(37.2, 0.5)
        # Improvement trend
        deterioration = np.linspace(0.2, 0, TIME_STEPS_PER_PATIENT)

    for t in range(TIME_STEPS_PER_PATIENT):
        timestamp = admission_time + timedelta(hours=t)
        deteri = deterioration[t]

        hr = hr_base + deteri * 20 + np.random.normal(0, 5)
        bp = bp_base - deteri * 15 + np.random.normal(0, 4)
        spo2 = spo2_base - deteri * 5 + np.random.normal(0, 1)
        rr = rr_base + deteri * 6 + np.random.normal(0, 2)
        temp = temp_base + deteri * 0.8 + np.random.normal(0, 0.2)

        # Clamp to physiological ranges
        hr = np.clip(hr, 30, 200)
        bp = np.clip(bp, 40, 200)
        spo2 = np.clip(spo2, 60, 100)
        rr = np.clip(rr, 5, 60)
        temp = np.clip(temp, 34, 42)

        # Introduce random missing values (~10%)
        if random.random() < 0.10:
            hr = np.nan
        if random.random() < 0.08:
            bp = np.nan
        if random.random() < 0.06:
            spo2 = np.nan
        if random.random() < 0.09:
            rr = np.nan
        if random.random() < 0.07:
            temp = np.nan

        records.append({
            'record_id': f"{patient_id}_{t}",
            'patient_id': patient_id,
            'timestamp': timestamp,
            'heart_rate': round(hr, 2) if not np.isnan(hr) else np.nan,
            'blood_pressure': round(bp, 2) if not np.isnan(bp) else np.nan,
            'oxygen_saturation': round(spo2, 2) if not np.isnan(spo2) else np.nan,
            'respiratory_rate': round(rr, 2) if not np.isnan(rr) else np.nan,
            'body_temperature': round(temp, 2) if not np.isnan(temp) else np.nan,
        })

    return records

def generate_dataset():
    print("Generating ICU dataset...")

    # Generate patient table
    patients = []
    all_vitals = []

    mortality_rate = 0.30  # 30% mortality rate (realistic ICU stat)

    for i in range(1, NUM_PATIENTS + 1):
        outcome = 1 if i <= int(NUM_PATIENTS * mortality_rate) else 0
        age = random.randint(18, 90)
        gender = random.choice(['Male', 'Female'])
        admission_time = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))

        patients.append({
            'patient_id': i,
            'age': age,
            'gender': gender,
            'admission_time': admission_time,
            'outcome_flag': outcome
        })

        vitals = generate_patient_vitals(i, outcome)
        all_vitals.extend(vitals)

    df_patients = pd.DataFrame(patients)
    df_vitals = pd.DataFrame(all_vitals)

    os.makedirs('dataset', exist_ok=True)
    df_patients.to_csv('dataset/patients.csv', index=False)
    df_vitals.to_csv('dataset/vital_signs.csv', index=False)

    print(f"✓ Patients: {len(df_patients)} records")
    print(f"✓ Vital Signs: {len(df_vitals)} records")
    print(f"✓ Mortality Rate: {df_patients['outcome_flag'].mean()*100:.1f}%")
    print(f"✓ Missing Values in vitals: {df_vitals.isnull().sum().sum()} cells")
    print("Dataset saved to dataset/patients.csv and dataset/vital_signs.csv")

if __name__ == "__main__":
    generate_dataset()

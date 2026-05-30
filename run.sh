#!/bin/bash
# ICU Project - One-click setup and run script

echo "========================================"
echo "  ICU Mortality Risk Prediction System  "
echo "========================================"

echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "[2/3] Generating dataset..."
cd dataset && python generate_dataset.py && cd ..

echo ""
echo "[3/3] Training models (this may take 10-20 mins)..."
cd backend && python train_models.py && cd ..

echo ""
echo "========================================"
echo "  Training Complete!"
echo "  Starting API server..."
echo "  Open frontend/index.html in browser"
echo "========================================"
cd backend && python app.py

---
title: Chest X-ray Diagnostic Assistant
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 🫁 Chest X-ray Diagnostic Assistant

AI-powered chest X-ray analysis system combining:
- **DenseNet-121** for multi-label disease classification
- **Grad-CAM** for explainable AI visualizations
- **LLM** for automated clinical report generation

## Features
- Upload chest X-ray images (JPG, PNG)
- Get predictions for 5 conditions:
  - Cardiomegaly
  - Edema
  - Consolidation
  - Atelectasis
  - Pleural Effusion
- View attention heatmaps (Grad-CAM)
- Generate clinical summaries

## Usage
1. Upload a chest X-ray image
2. Click "Analyze X-ray"
3. View predictions, heatmaps, and AI-generated report

⚠️ **Disclaimer**: This is a research prototype. Not for clinical use.

## Model Details
- Architecture: DenseNet-121
- Dataset: CheXpert
- Framework: PyTorch
- Explainability: Grad-CAM++

## Developer
Created as part of an AI medical imaging project.
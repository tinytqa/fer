---
title: FER Web App
emoji: 🎭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app_gradio.py
pinned: false
---

# Facial Emotion Recognition (FER)

This repository contains a PyTorch ResNet50 model for Facial Emotion Recognition. 
It supports both local desktop inference and Web UI deployment on Hugging Face Spaces via Gradio.

## Chạy trên máy cá nhân (Desktop)
```bash
pip install -r requirements.txt
python app.py
```

## Chạy dưới dạng Web App (Gradio)
```bash
python app_gradio.py
```
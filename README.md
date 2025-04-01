# Realtime Micro-Expression Regression Demo

This repository demonstrates a real-time, regression-based valence-arousal estimation system built upon the model proposed in our [AAAI 2025 paper](https://arxiv.org/abs/2502.09993).
The original classification framework has been extended to a regression task to estimate subtle facial expression dynamics (e.g., valence, arousal, expression intensity).

---

## Description

- Based on the NLA (Navigating Label Ambiguity) model, modified for regression
- Receives live webcam input
- Detects faces in real time and estimates subtle expression states continuously
- Suitable for applications such as emotional monitoring or human-computer interaction

---

## Architecture

<img width="1258" alt="Image" src="https://github.com/user-attachments/assets/be10babc-55a9-46b9-b9f3-4d5f14fd2acb" />

---

## Demo Video

<video src="https://github.com/user-attachments/assets/6746d16f-fdab-46d8-bbe5-3cb256d124b8" controls width="600">
  Your browser does not support the video tag.
</video>

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/realtime_emotion_regression
cd realtime_emotion_regression
python main.py

# Spatial–Frequency Domain Aggregation Upsampling for Pansharpening

This is the official implementation of the paper **"Spatial–Frequency Domain Aggregation Upsampling for Pansharpening"**.

## 📌 Project Overview

This repository provides the implementation of the **SFAU (Spatial–Frequency domain Aggregation Upsampling)** module proposed in the paper. The SFAU module can be easily plugged into existing deep learning frameworks to enhance the performance of pansharpening models.

---

## 🧠 Model

- `model/SFAU.py`: Contains the implementation of the SFAU module. It can be directly imported and integrated into your own networks as a plug-and-play upsampling component.

---

## 🏋️‍♂️ Training

- `train_pannet_model.py`: Provides a training pipeline using PanNet as the baseline model. You can replace the upsampling module with SFAU or other methods to train and compare performance.

---

## 📈 Evaluation

- `cal_pannet_metric.py`: Computes pansharpening quality metrics on the test dataset for trained models. This includes commonly used indicators such as PSNR, SSIM, SAM, etc.

---

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

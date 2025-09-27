# Fingerprint-Liveness-Detection

This repository contains Python code for a **Biometric Security project** focused on **fingerprint liveness detection**. The project leverages a **modified ResNet-18 architecture with Fast Fourier Convolution (FFC) blocks** to distinguish between **live** and **fake** fingerprints using the **LivDet 2015 dataset**.

---

## 📖 Project Overview

Fingerprint recognition systems are vulnerable to **presentation attacks (spoofing)** using artificial fingerprints. This project addresses this security challenge by designing a **binary classification model** capable of detecting whether a fingerprint is **live (genuine)** or **fake (spoofed)**.

The workflow consists of three key stages:

### 1. 🖼️ Image Pre-processing

* Raw fingerprint images are **cropped to the region of interest (ROI)**.
* Resized to **224×224 pixels** to standardize input for the neural network.
* This ensures consistent input representation across different fingerprint sensors.

### 2. 🧠 Model Architecture

* **Base model:** ResNet-18.
* **Modification:** Standard convolutional layers replaced with **Fast Fourier Convolution (FFC) blocks**.
* **Advantage:** FFC splits feature processing into:

  * **Local spatial features**
  * **Global frequency features**
* This helps capture subtle textural differences between live and fake fingerprints more effectively than traditional CNNs.

### 3. 🎯 Training & Evaluation

* **Loss Function:** `BCEWithLogitsLoss` (binary classification).
* **Optimizer:** Adam.
* **Training Strategy:**

  * Trained on a **combined dataset of live and fake fingerprints** from multiple sensors.
  * Evaluated on **individual sensor datasets** to measure **generalization performance**.

---

## 📊 Results

* The modified **FFC-ResNet18** achieved strong performance in detecting fake fingerprints.
* Demonstrated **better generalization across multiple sensors** compared to standard CNN baselines.
  

# Fingerprint-Liveness-Detection

This repository contains Python code, primarily for a Biometric Security project focused on fingerprint liveness detection. The project utilizes a modified ResNet model with Fast Fourier Convolution (FFC) blocks to distinguish between "live" and "fake" fingerprints from the LivDet 2015 dataset.

# Project Overview

The core of this project is a binary classification model designed to solve the problem of presentation attacks (spoofing) in fingerprint recognition systems. The workflow is divided into three main stages:

Image Pre-processing: The raw fingerprint images are cropped to the region of interest (ROI) and then resized to a uniform 224x224 pixel dimension. This step is crucial for standardizing the input for the neural network.

Model Architecture: A ResNet-18 model is modified by replacing standard convolutional layers with Fast Fourier Convolution (FFC) blocks. This technique, which splits feature processing into local spatial and global frequency components, is intended to improve the model's ability to detect subtle textural differences between live and fake fingerprints.

Training and Evaluation: The model is trained on a combined dataset of live and fake fingerprints from multiple sensors and then evaluated on individual sensor datasets to assess its generalization capabilities. The training process uses a BCEWithLogitsLoss criterion and an Adam optimizer.

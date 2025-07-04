#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import argparse

# Allow loading incomplete images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------
# Spectral Convolution Layer
# -----------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        h_modes = min(self.modes1, x_ft.shape[2])
        w_modes = min(self.modes2, x_ft.shape[3])

        out_ft[:, :, :h_modes, :w_modes] = self.compl_mul2d(x_ft[:, :, :h_modes, :w_modes], self.weights1[:, :, :h_modes, :w_modes])
        out_ft[:, :, -h_modes:, :w_modes] = self.compl_mul2d(x_ft[:, :, -h_modes:, :w_modes], self.weights2[:, :, -h_modes:, :w_modes])

        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x

# -----------------------
# FFC Residual Block
# -----------------------
class FFC_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=12, modes2=12):
        super(FFC_ResBlock, self).__init__()
        self.conv1 = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.res_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.res_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + residual)

# -----------------------
# FFT-Enhanced ResNet-34
# -----------------------
class FFTResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(FFTResNet34, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(FFC_ResBlock, 64, 64, 3)
        self.layer2 = self._make_layer(FFC_ResBlock, 64, 128, 4)
        self.layer3 = self._make_layer(FFC_ResBlock, 128, 256, 6)
        self.layer4 = self._make_layer(FFC_ResBlock, 256, 512, 3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, in_channels, out_channels, blocks):
        layers = [block(in_channels, out_channels)]
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def extract_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)

# -----------------------
# Utility Functions
# -----------------------
def load_image_paths(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

def extract_label_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    parts_lower = [p.lower() for p in parts]
    if 'live' in parts_lower:
        return 1
    elif 'fake' in parts_lower:
        return 0
    else:
        raise ValueError(f"Label not found in path: {path}")

def load_and_preprocess_image(img_path, transform):
    try:
        image = Image.open(img_path).convert('RGB')
        return transform(image)
    except Exception as e:
        print(f"[WARN] Failed to load image: {img_path} — {e}")
        return None

def extract_batch_features(paths, model, transform, device):
    features, valid_paths = [], []
    for path in paths:
        img_tensor = load_and_preprocess_image(path, transform)
        if img_tensor is None:
            continue
        img_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.extract_features(img_tensor).cpu().numpy().flatten()
        features.append(feat)
        valid_paths.append(path)
    return np.array(features), valid_paths

# -----------------------
# Argument Parser
# -----------------------
parser = argparse.ArgumentParser(description="Fingerprint Feature Extraction using FFT-ResNet34")
parser.add_argument("--template", type=str, required=True, help="Path to template image list file (txt)")
parser.add_argument("--probe", type=str, required=True, help="Path to probe image list file (txt)")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

model = FFTResNet34(num_classes=2).to(device)
model.eval()

# template_paths = load_image_paths("templateimagesfile.txt")
# probe_paths = load_image_paths("probeimagesfile.txt")
template_paths = load_image_paths(args.template)
probe_paths = load_image_paths(args.probe)

template_labels = [extract_label_from_path(p) for p in template_paths]

probe_features, _ = extract_batch_features(probe_paths, model, transform, device)
template_features, _ = extract_batch_features(template_paths, model, transform, device)

# -----------------------
# XGBoost Classifier
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(template_features, template_labels, test_size=0.2, stratify=template_labels)

xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
xgb_clf.fit(X_train, y_train)

probe_preds = xgb_clf.predict(probe_features)
with open("probe_predictions.txt", "w") as f:
    for path, pred in zip(probe_paths, probe_preds):
        f.write(f"{path},{pred}\n")

val_preds = xgb_clf.predict(X_val)
acc = accuracy_score(y_val, val_preds)
print(f"XGBoost Validation Accuracy: {acc*100:.2f}%")

np.save("template_features.npy", template_features)
np.save("probe_features.npy", probe_features)

match_scores = cosine_similarity(probe_features, template_features).max(axis=1)
liveness_scores = xgb_clf.predict_proba(probe_features)[:, 1] * 100
ims_scores = (liveness_scores + match_scores * 100) / 2

np.savetxt("IMSoutputfile.txt", ims_scores, fmt="%.4f")
print("Saved IMS scores to 'IMSoutputfile.txt'")
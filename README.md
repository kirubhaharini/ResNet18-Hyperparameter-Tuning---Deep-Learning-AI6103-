# ResNet-18 Hyperparameter Tuning on CIFAR-10
---

## 📋 Overview

This project systematically investigates the impact of hyperparameter tuning on ResNet-18 performance for CIFAR-10 image classification. Through controlled experimentation, four key aspects are evaluated:

1. **Learning Rate Selection** (0.001, 0.01, 0.1)
2. **Learning Rate Scheduling** (Constant vs Cosine Annealing)
3. **Weight Decay Regularization** (5×10⁻⁴ vs 1×10⁻²)
4. **Custom Batch Normalization** (Gradient flow analysis)

---

## 🎯 Key Results

| Configuration | Validation Accuracy | Validation Loss | Notes |
|--------------|---------------------|-----------------|-------|
| Initial (LR=0.01, 15 epochs) | 85.82% | 0.4320 | Early stopping |
| **Baseline (LR=0.01, 300 epochs)** | 92.38% | 0.5071 | Fair comparison |
| + Cosine Annealing | 92.59% | 0.4547 | +0.21% |
| + Weight Decay 5×10⁻⁴ | 94.12% | 0.2448 | +1.74% |
| **+ Weight Decay 1×10⁻²** | **95.06%** | **0.1919** | **+2.68%** |
| Custom BatchNorm | 21.65% | 1.9775 | Failed |

**Final Test Results:**
- **Test Accuracy:** 95.10%
- **Test Loss:** 0.1846

**Optimal Configuration:**
- Learning Rate: 0.01
- Scheduler: Cosine Annealing
- Weight Decay: 1×10⁻²
- **Improvement over baseline:** +2.68% validation accuracy

---

## 🏗️ Architecture: ResNet-18

ResNet-18 is a residual neural network with skip connections that enable training of deep networks by solving the vanishing gradient problem.

**Key Components:**
- **Initial Conv:** 3×3, 64 filters, stride 1
- **4 Residual Stages:**
  - Stage 1: 2 blocks, 64 channels, 32×32 spatial resolution
  - Stage 2: 2 blocks, 128 channels, 16×16 (stride 2)
  - Stage 3: 2 blocks, 256 channels, 8×8 (stride 2)
  - Stage 4: 2 blocks, 512 channels, 4×4 (stride 2)
- **Global Average Pooling:** 4×4 → 512×1×1
- **Fully Connected:** 512 → 10 classes

**Parameters:** ~11M (vs VGG-18's ~138M)

---

## 📊 Experimental Setup

### Dataset: CIFAR-10
- **Total Images:** 60,000 (32×32 RGB)
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Split:** 40,000 train / 10,000 validation / 10,000 test

### Data Augmentation
- Random horizontal flip
- Random crop with 4-pixel padding
- Normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]

### Training Configuration
- **Optimizer:** SGD with momentum 0.9
- **Batch Size:** 128
- **Loss Function:** Cross-Entropy
- **Hardware:** NVIDIA GPU
- **Epochs:** 15 (learning rate tuning), 300 (all other experiments)

---

## 🔬 Experiments

### 1. Learning Rate Selection (15 epochs)

Evaluated three learning rates to find optimal convergence speed vs stability:

| Learning Rate | Train Acc | Val Acc | Train Loss | Val Loss |
|--------------|-----------|---------|------------|----------|
| 0.001 | 85.93% | 80.28% | 0.4049 | 0.5868 |
| **0.01** | **91.05%** | **85.82%** | **0.2558** | **0.4320** |
| 0.1 | 87.20% | 84.54% | 0.3658 | 0.4607 |

**Finding:** LR=0.01 provides optimal balance between convergence speed and stability.

---

### 2. Learning Rate Scheduler (300 epochs)


Compared constant learning rate vs cosine annealing schedule:

**Cosine Annealing Formula:**
```
η_t = η_min + 0.5(η_max - η_min)(1 + cos(T_cur/T_max × π))
```

| Configuration | Train Acc | Val Acc | Train Loss | Val Loss |
|--------------|-----------|---------|------------|----------|
| No Scheduler | 99.95% | 92.38% | 0.0016 | 0.5071 |
| **Cosine Annealing** | **100.00%** | **92.59%** | **0.0001** | **0.4547** |

**Finding:** Cosine annealing reduces validation loss by 10.3% through gradual learning rate decay, enabling fine-grained late-stage optimization.

---

### 3. Weight Decay Regularization (300 epochs)

Evaluated L2 regularization strength:

| Weight Decay | Train Acc | Val Acc | Train Loss | Val Loss |
|-------------|-----------|---------|------------|----------|
| 5×10⁻⁴ | 100.00% | 94.12% | 0.0012 | 0.2448 |
| **1×10⁻²** | **100.00%** | **95.06%** | **0.0229** | **0.1919** |

**Finding:** Stronger regularization (1×10⁻²) improves validation accuracy by 0.94% and reduces validation loss by 21.6%, demonstrating effective overfitting prevention despite higher training loss.

---

### 4. Custom Batch Normalization (300 epochs)

Implemented custom BatchNorm where mean and variance statistics are **detached from gradient computation** using PyTorch's `detach()` function.

**Results:**

| Configuration | Train Acc | Val Acc | Train Loss | Val Loss |
|--------------|-----------|---------|------------|----------|
| **Custom BN** | 21.95% | 21.65% | 1.9707 | 1.9775 |

**Observation:** Complete training failure with:
- Severe oscillations (loss spikes to 80+)
- Accuracy stuck at ~22% (near random guessing for 10 classes)
- Highly unstable gradient updates

**Finding:** Gradient flow through batch statistics is **essential** for stable optimization in deep networks. Detaching statistics causes:
1. **Gradient Instability:** Uncontrolled updates leading to loss spikes
2. **Unstable Optimization:** Irregular loss landscape with overshooting
3. **Broken Gradient Coordination:** Conflicting updates across samples

---

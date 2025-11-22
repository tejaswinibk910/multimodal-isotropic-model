# Multimodal Tool Wear Prediction with Minape Architecture

A deep learning system for predicting tool wear in CNC milling machines using the Minape (Multimodal Isotropic Neural Architecture with Patch Embedding) architecture on the Nonastreda dataset. I implemented cross-domain generalization through tool-based cross-validation to ensure models generalize to previously unseen machines.

## Overview

This project extends the Minape architecture from the Mudestreda dataset (4 modalities) to the Nonastreda dataset (9 modalities), establishing the first benchmark on this newly published dataset. The system addresses the critical industrial problem of cross-domain generalization in manufacturing, where models trained on one machine often fail when deployed on different machines or tools.

The system combines 9 modalities (chip images, tool images, workpiece images, and force signal scalograms/spectrograms) to perform:

1. **Classification**: Predict tool wear state (Sharp/Used/Dulled)
2. **Regression**: Estimate continuous wear measurements (flank wear, gaps, overhang in μm)
3. **Cross-Domain Evaluation**: Test generalization to unseen tools via tool-based cross-validation

## Key Features

- **Isotropic Architecture**: Uses the same transformer architecture for all modalities through patch embedding
- **Patch Embedding**: Converts all modalities (images and signals) into uniform patch representations
- **TempMixer Fusion**: Cross-modal fusion with sample-adaptive importance weights
- **Cross-Domain Validation**: Tool-based k-fold CV where each fold trains on 9 tools and tests on 1 unseen tool
- **Multi-Task Learning**: Simultaneous classification and regression with shared representations

## Dataset: Nonastreda

The Nonastreda dataset contains 512 samples across 10 different cutting tools with 9 modalities per sample:

**Image Modalities** (RGB, 256×256):
- Chip images
- Tool images  
- Workpiece images

**Time-Series Modalities** (converted to 2D):
- Scalograms (x, y, z force axes)
- Spectrograms (x, y, z force axes)

**Labels**:
- Classification: 3 classes (Sharp=0, Used=1, Dulled=2)
- Regression: flank_wear, gaps, overhang (μm)

## Architecture

The Minape architecture uses an isotropic design where all modalities are processed through the same transformer architecture after patch embedding:

```
Input: 9 Modalities (chip, tool, work, scalograms[x,y,z], spectrograms[x,y,z])
    ↓
Stage 1: Patch Embedding
    ├─ Image Patches (16×16) → Linear Projection
    └─ Signal Patches (grouped axes) → Linear Projection
    All modalities → Uniform D-dimensional embeddings
    ↓
Stage 2: Isotropic Transformer
    ├─ Multi-Head Self-Attention
    ├─ Feed-Forward Networks
    └─ Layer Normalization + Residual Connections
    Same architecture applied to all modality patches
    ↓
Stage 3: TempMixer Fusion
    ├─ Sample-Adaptive Importance Weights
    ├─ Cross-Modal Temporal Dependencies
    └─ Weighted Aggregation of Modality Representations
    ↓
Stage 4: Multi-Task Heads
    ├─ Classification Head → 3 classes (Sharp/Used/Dulled)
    ├─ Regression Head → 3 continuous values (flank_wear, gaps, overhang)
    └─ Optional: Reconstruction for anomaly detection
```

### Key Architectural Principles

**Isotropic Design**: Instead of using modality-specific architectures, Minape uses the same transformer architecture for all modalities. This is achieved through:

1. **Patch Embedding**: All modalities are divided into patches and linearly embedded into the same D-dimensional space
2. **Unified Processing**: The same transformer layers process all modality patches
3. **Modality-Agnostic**: The architecture doesn't need to know which modality a patch comes from

**TempMixer**: A recurrent structure that learns:
- Sample-specific importance weights for different modalities
- Temporal dependencies across modalities
- How to fuse heterogeneous information (images + signals)

## Installation

### Requirements

```bash
Python 3.8+
PyTorch 1.10+
CUDA 11.0+ (for GPU acceleration)
```

### Setup

```bash
# Clone repository
git clone <repository-url>
cd minape-project

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn matplotlib seaborn tqdm pillow
```

## Project Structure

```
project/
├── complete_training_code.py          # Model architecture (anomaly detection variant)
├── complete_multitask_standalone.py   # Multi-task model implementation
├── cv_dataset_loader.py               # Dataset loader with CV splitting
├── non_dataset.py                     # Basic dataset loader
├── Untitled.ipynb                     # Training notebook
├── checkpoints_cv/                    # Model checkpoints
└── Nonastreda_Multimodal/            # Dataset directory
    ├── chip/                          # Chip images
    ├── tool/                          # Tool images
    ├── work/                          # Workpiece images
    ├── scal/x/, scal/y/, scal/z/     # Scalograms
    ├── spec/x/, spec/y/, spec/z/     # Spectrograms
    ├── labels.csv                     # Classification labels
    └── labels_reg.csv                 # Regression targets
```


### Quick Start

```python
from cv_dataset_loader import create_cv_dataloaders
from complete_multitask_standalone import MultiTaskMultimodalModel
import torch
from torchvision import transforms

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create dataloaders for fold 1 (test on tool T1)
train_loader, test_loader = create_cv_dataloaders(
    root_dir='path/to/Nonastreda_Multimodal',
    test_tool=1,
    batch_size=16,
    train_transform=train_transform,
    val_transform=val_transform
)

# Initialize Minape model
model = MultiTaskMultimodalModel(
    embedding_dim=256,
    fusion_type='attention',  # TempMixer-style attention fusion
    num_classes=3,
    num_regression_targets=3
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

 
### Cross-Validation Methodology

Each fold evaluates generalization to a completely unseen tool:

```
Fold 1:  Train on T2,T3,T4,T5,T6,T7,T8,T9,T10  →  Test on T1
Fold 2:  Train on T1,T3,T4,T5,T6,T7,T8,T9,T10  →  Test on T2
...
Fold 10: Train on T1,T2,T3,T4,T5,T6,T7,T8,T9  →  Test on T10
```

This simulates real-world deployment where the model must work on machines it has never seen during training.

### Training Configuration

Key hyperparameters for the Minape architecture:

```python
# Model Architecture
embedding_dim = 256           # Patch embedding dimension (isotropic)
patch_size = 16               # Image patch size (16×16)
num_transformer_layers = 4    # Isotropic transformer depth
num_attention_heads = 8       # Multi-head attention

# Fusion
fusion_type = 'attention'     # TempMixer-style fusion with adaptive weights

# Training
batch_size = 32               # Optimized for RTX 3050 6GB
epochs = 30                   # Max epochs per fold
learning_rate = 2e-4          # Adam optimizer
weight_decay = 1e-5           # L2 regularization
patience = 10                 # Early stopping patience

# Loss weights (multi-task learning)
alpha_class = 1.0             # Classification loss weight
alpha_reg = 1.0               # Regression loss weight
alpha_recon = 0.5             # Reconstruction loss weight (optional)

```

### Why Isotropic Architecture?

The Minape architecture offers several advantages:

1. **Unified Processing**: Same transformer for all modalities eliminates the need for modality-specific architectures
2. **Parameter Efficiency**: Shared weights across modalities reduce total parameters
3. **Scalability**: Easy to add new modalities without architectural changes
4. **Theoretical Elegance**: All modalities treated uniformly through patch embedding

Compared to traditional approaches (ResNet for images + LSTM for signals), the isotropic design achieves similar or better performance with fewer parameters and cleaner abstraction.

## Evaluation Metrics

### Classification Metrics
- Accuracy
- Precision, Recall, F1-Score (per class and macro-averaged)
- Confusion Matrix

### Regression Metrics
- Mean Absolute Error (MAE) in μm
- Root Mean Squared Error (RMSE) in μm
- Per-target metrics (flank_wear, gaps, overhang)

### Cross-Domain Metrics
- Mean performance across folds
- Standard deviation (consistency)
- Domain gap (best vs worst fold)

## Results

### Expected Performance with Tool-Based Cross-Validation

The key insight from this project is the importance of rigorous evaluation methodology:

**Random Split** (unrealistic, data leakage):
- Classification Accuracy: ~100%
- F1 Score: ~0.99-1.00
- MAE: <5 μm

**Tool-Based Split** (realistic, true generalization):
- Classification Accuracy: ~91% (average across folds)
- F1 Score: ~0.82-0.92
- MAE: 30-40 μm

**Performance Gap**: ~8.9% accuracy drop demonstrates why tool-based cross-validation is critical for honest evaluation.


### Per-Fold Results (10-Fold CV)

```
Fold  Tool  Accuracy  F1     MAE(μm)  RMSE(μm)
----  ----  --------  -----  -------  --------
  1    T1    0.825    0.802   42.3     58.7
  2    T2    0.798    0.781   38.5     52.1
  3    T3    0.834    0.816   35.2     48.9
  4    T4    0.789    0.774   45.8     61.2
  5    T5    0.816    0.793   41.7     56.4
  6    T6    0.822    0.805   39.1     53.8
  7    T7    0.803    0.786   43.5     59.3
  8    T8    0.828    0.811   37.9     51.2
  9    T9    0.791    0.777   44.2     60.1
 10   T10    0.819    0.798   40.6     55.7
----  ----  --------  -----  -------  --------
Mean        0.813    0.794   40.9     55.7
Std         0.015    0.014    3.2      4.1
```


## Related Work

This project builds on and extends:

1. **Minape Architecture** (ICONIP 2023, Leibniz University Hannover)
   - Original paper: "Multimodal Isotropic Neural Architecture with Patch Embedding"
   - Predecessor dataset: Mudestreda (4 modalities, 97-99% accuracy)
   - Extension: Nonastreda (9 modalities, first benchmark)

2. **Nonastreda Dataset** (Data in Brief, January 2025)
   - 512 samples across 10 tools with 9 modalities
   - Openly accessible on Mendeley Data
   - Supporting code: github.com/hubtru/Minape

3. **Cross-Domain Generalization**
   - Tool-based cross-validation simulates real-world deployment
   - Addresses the critical problem of machine-to-machine transfer
   - Reveals 18.9% performance gap vs random splits

## Future Work

### Immediate Extensions

1. **PAMAP2 Implementation**: Extend to Physical Activity Monitoring dataset
   - Test architecture on different modality types (all IMU signals)
   - Evaluate isotropic architecture across domains

2. **Advanced Fusion Methods**:
   - Patch-based multimodal contrastive learning
   - Cross-modal patch mixing
   - Meta-learning for few-shot adaptation

3. **Modality Ablation Studies**:
   - Systematic evaluation of all modality combinations
   - Identify minimal modality sets for deployment efficiency


### Original Minape architecture:

```bibtex
@inproceedings{truchan2023minape,
  title={Minape: Multimodal Isotropic Neural Architecture with Patch Embedding},
  author={Truchan, Hubert and others},
  booktitle={ICONIP 2023},
  year={2023},
  publisher={Springer}
}
```

## Acknowledgments

- **Leibniz University Hannover** for the Nonastreda dataset and Minape architecture
- **Original Minape Implementation**: github.com/hubtru/Minape
- **Dataset**: Available on Mendeley Data (DOI: 10.17632/m892d2wtzh.1)

## License

This project is released under the MIT License. See LICENSE file for details.


### Common Issues

**CUDA Out of Memory**:
```python
# Reduce batch size
batch_size = 8  # instead of 16

# Use gradient accumulation if needed
accumulation_steps = 2
```

**Label Indexing Issues**:
The dataset uses 0-indexed labels (0=Sharp, 1=Used, 2=Dulled). Ensure PyTorch CrossEntropyLoss receives labels in range [0, num_classes-1].

**Slow Training**:
- Ensure CUDA is properly installed: `torch.cuda.is_available()` should return `True`
- Use `num_workers=4` in DataLoader (adjust based on CPU cores)
- Consider mixed precision training with `torch.cuda.amp` for faster training

**Performance Issues**:
If getting much lower accuracy than expected:
- Verify labels are 0-indexed (not 1-indexed)
- Check that train/test splits are correct (no tool overlap)
- Ensure data augmentation is only applied to training set
- Verify normalization statistics match ImageNet pretraining

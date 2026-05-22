# PekoraNet

> A Lightweight Hierarchical CNN Framework for CSI Feedback Compression in Massive MIMO Systems

---

# Overview

PekoraNet is a lightweight deep learning framework designed to reduce CSI (Channel State Information) feedback overhead in massive MIMO systems.

Unlike conventional CSI compression methods that rely heavily on shallow fully-connected architectures, PekoraNet introduces:

- Hierarchical CNN-based feature extraction
- Residual reconstruction refinement
- Spatial redundancy-aware compression
- Stable low-compression-ratio reconstruction

The model is specifically designed for:

- practical deployment feasibility
- lightweight inference
- robust reconstruction under aggressive compression ratios

---

# Key Features

## Hierarchical CSI Compression

PekoraNet exploits the strong spatial correlation of CSI matrices using multi-stage CNN downsampling.

```text
CSI Tensor
 → CNN Feature Extraction
 → Hierarchical Downsampling
 → Latent Compression
```

This enables semantic-aware CSI representation learning instead of naive flatten-based compression.

---

## Residual Reconstruction Refinement

The decoder integrates residual refinement blocks to improve reconstruction fidelity under low compression ratios.

Advantages:

- improved gradient flow
- stable convergence
- better high-frequency channel reconstruction
- lower NMSE degradation

---

## Lightweight Architecture

Designed for realistic UE-side deployment:

- low parameter count
- moderate FLOPs
- stable training
- practical inference latency

---

## Stable Training Pipeline

PekoraNet includes several modern training stabilization techniques:

- AdamW optimizer
- CosineAnnealingLR scheduler
- Gradient clipping
- Mixed Precision Training (AMP)
- BatchNorm
- LeakyReLU activation

---

# Architecture

## Encoder

```python
Conv2d(2 → 16, stride=2)
Conv2d(16 → 32, stride=2)
Conv2d(32 → 32)
Flatten
Linear → latent_dim
```

---

## Decoder

```python
Linear(latent_dim → feature map)
ConvTranspose2d
ConvTranspose2d
ResidualBlock
Conv2d → reconstructed CSI
```

---

# Motivation

CSI feedback overhead becomes a critical bottleneck in massive MIMO systems.

Existing approaches such as CsiNet suffer from:

- shallow feature extraction
- unstable low-CR reconstruction
- weak spatial hierarchy modeling
- limited practical deployment considerations

PekoraNet addresses these limitations through hierarchical CNN compression and residual reconstruction refinement.

---

# Compression Ratios

Supported compression ratios:

- CR = 1/4
- CR = 1/8
- CR = 1/16
- CR = 1/32
- CR = 1/64

Special focus is placed on robust reconstruction under extremely low feedback budgets.

---

# Dataset

The framework supports:

- Indoor CSI datasets
- Outdoor CSI datasets
- Joint indoor/outdoor training

Example scenarios:

```python
scenario = "indoor"
scenario = "outdoor"
scenario = "joint"
```

---

# Training

## Example

```bash
python train.py \
  --scenario indoor \
  --cr 32 \
  --epochs 200
```

---

## Recommended Environment

- Python 3.10+
- PyTorch 2.x
- CUDA 11.x+

---

# Evaluation Metrics

Main evaluation metrics:

- NMSE (Normalized Mean Square Error)
- Cosine Similarity
- Reconstruction Loss

Future extensions may include:

- Spectral Efficiency
- Beamforming Gain

---

# Comparison with CsiNet

| Feature | CsiNet | PekoraNet |
|---|---|---|
| Compression Method | FC-based | Hierarchical CNN |
| Spatial Correlation Modeling | Limited | Strong |
| Residual Reconstruction | Minimal | Yes |
| Low-CR Robustness | Moderate | Improved |
| Training Stabilization | Basic | Advanced |
| Deployment Feasibility | Moderate | Practical |

---

# Future Work

Planned future extensions:

- Attention-enhanced CSI encoding
- Quantization-aware training
- Adaptive compression ratio support
- Temporal CSI modeling
- Beamforming-aware optimization
- Semantic CSI feedback

---

# Research Direction

PekoraNet aims to bridge the gap between:

```text
high-performance CSI reconstruction
and
practical wireless deployment
```

The framework focuses on:

- robustness
- lightweight design
- deployment feasibility
- stable reconstruction

rather than excessive architectural complexity.

---

# Citation

If you use PekoraNet in your research or projects, please consider citing this repository.

Formal publication information will be added in a future release.
```

---

# License

MIT License

---

# Acknowledgements

Inspired by:

- CsiNet
- CRNet
- Deep learning-based CSI compression research
- Massive MIMO feedback optimization studies

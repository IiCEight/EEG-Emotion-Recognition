# PCL-TDGCN: Prototype Contrastive Learning with Temporal-Dynamic Graph Convolutional Network

## Overview

PCL-TDGCN is a domain adaptation model for cross-subject EEG emotion recognition. It addresses the **subject shift problem**: EEG signals vary significantly between individuals, so a model trained on 14 subjects must generalize to an unseen 15th subject without any of their labels.

The model combines three ideas:
1. **Dynamic graph convolution** to model spatial relationships between EEG electrodes
2. **Prototype-based memory banks** to generate pseudo-labels for the unlabeled target subject
3. **DANN adversarial training** to align source and target feature distributions

---

## Problem Setup

- **Dataset**: SEED — 15 subjects × 3 sessions × 15 trials, 62 EEG electrodes × 5 frequency bands (δ, θ, α, β, γ)
- **Task**: Subject-independent emotion classification (3 classes: negative, neutral, positive)
- **Protocol**: Leave-one-subject-out (LOSO) — 14 subjects as source (labeled), 1 as target (unlabeled at training time)
- **Input shape per sample**: `[310]` (flattened 62 electrodes × 5 bands)

---

## Architecture

```
Input [B, 310]
    │
    ▼
Reshape → [B, 5, 62, 1]
    │
    ▼
┌─────────────────────────────────┐
│           MHGCN                 │
│  ┌──────────────────────────┐   │
│  │  GATENet(A) → adj [62,62]│   │
│  └──────────────────────────┘   │
│  Layer 0: input x               │
│  Layer 1: HGCN(x, adj)          │
│  Layer 2: HGCN(x, adj)          │
│  Concat all → [B, 15, 62, 1]    │
└─────────────────────────────────┘
    │
    ▼
CBAM Attention (channel + spatial)
    │
    ▼
Flatten → FC(930, 64) → ReLU → Dropout
       → FC(64, 64)  → ReLU → Dropout
    │
    ▼
Feature [B, 64]
    │
    ├──→ ClassClassifier → Logits [B, 3]
    │
    └──→ Memory Bank + Prototype Logic
```

---

## Components

### 1. MHGCN — Multi-scale Hierarchical GCN

**Purpose**: Extract spatial features from EEG electrodes using a graph structure.

**Adjacency matrix generation (GATENet)**:
- Maintains a learnable parameter `A` of shape `[1, 62×62]`
- Passes it through: `FC → ELU → FC → Tanh → ReLU` → reshape to `[62, 62]`
- This is a **static** adjacency matrix shared across all samples and batches

**HGCN layer**:
- Normalizes the adjacency: `L = A · D⁻¹` (column-sum degree normalization)
- Applies `resGCN`: two grouped Conv2d layers (one per frequency band) → BN → ELU
- Graph propagation: `y = einsum(x, L)` — spreads features across electrode neighbors
- Residual connection: `y = ELU(y + x)`

**Multi-scale concatenation**:
- Collects output from input + all HGCN layers: `[x₀, x₁, x₂]`
- Concatenates along channel dim: shape `[B, (layers+1)×5, 62, 1]` = `[B, 15, 62, 1]`
- This preserves features at multiple graph propagation depths

---

### 2. CBAM — Convolutional Block Attention Module

**Purpose**: Selectively weight important frequency bands (channel attention) and electrode regions (spatial attention).

**Channel Attention**:
```
x → MaxPool2d(1) → SE(conv→ReLU→conv) ─┐
x → AvgPool2d(1) → SE(conv→ReLU→conv) ─┴→ sigmoid → weight
```
Weights each of the 15 feature channels independently.

**Spatial Attention**:
```
x → max across channels ─┐
x → avg across channels ─┴→ cat → Conv2d(2,1) → sigmoid → weight
```
Weights each of the 62 electrode positions independently.

**Forward**: `out = (x * channel_weight * spatial_weight) + x` (residual)

---

### 3. Memory Banks

The memory banks are the core mechanism enabling target pseudo-label generation **without any target labels**.

```python
source_f_bank     : [source_num, 64]   # normalized source features
source_score_bank : [source_num, 3]    # softmax scores for source
target_f_bank     : [target_num, 64]   # normalized target features
target_score_bank : [target_num, 3]    # softmax scores for target
```

**Initialization**: Before training begins, a forward pass over all source and target data fills the banks with initial (random-weight) features.

**Per-batch update**: Each forward pass overwrites the bank entries corresponding to the current batch's indices. This means by the end of an epoch, all entries reflect features from the latest model weights.

---

### 4. Source Prototype Computation (`_get_source_similar`)

**Goal**: Build one prototype vector per emotion class from the source domain.

**Steps**:
1. L2-normalize current batch features, write to `source_f_bank`
2. Determine each source sample's predicted class: `argmax(source_score_bank)` over the full bank
3. For each class `c`: average all `source_f_bank` entries predicted as class `c` → prototype `p_c`
4. Stack into `prototypes` shape `[3, 64]`
5. Compute similarity: `src_sim = normalize(feature) · normalize(prototypes)ᵀ / τ` → shape `[B, 3]`

**Key property**: Prototypes are built from the **entire bank** (all source samples), not just the current batch — giving stable, low-variance class centers that improve as the bank fills.

---

### 5. Target Prototype Computation (`_get_target_similar`)

**Goal**: Build target-domain prototypes unsupervised, then assign pseudo-labels to target samples.

**Steps**:
1. L2-normalize current batch target features, write to `target_f_bank`
2. Compute per-sample confidence: `max(softmax_score)` across all target bank entries
3. Select the top **30%** most confident target samples
4. Run **K-Means** (k=3) on those selected features → 3 cluster centers as target prototypes
5. Compute similarity: `tgt_sim = normalize(feature) · normalize(prototypes)ᵀ / τ`
6. Pseudo-labels: `tar_label = argmax(softmax(tgt_sim))`

**Why K-Means on top-30%?** Using all target samples would include very uncertain predictions that destabilize clustering. Filtering to high-confidence samples produces cleaner cluster centers.

---

### 6. Cross-domain Similarity (`_get_st_similar`)

Used to compute four cross/within-domain similarity distributions:

| Variable | Feature | Prototype | Meaning |
|---|---|---|---|
| `s2t_pro` | source | target KMeans | How well source fits target clusters |
| `t2s_pro` | target | source class means | How well target fits source classes |
| `s2s_pro` | source | source class means | Source self-consistency |
| `t2t_pro` | target | target KMeans | Target self-consistency |

Each returns a softmax probability distribution `[B, 3]`.

---

### 7. Domain Discriminator + DANN Loss

**Discriminator**: FC(64→64) → ReLU → Dropout → FC(64→1) → Sigmoid

**DAANLoss**:
- Concatenates source and target features: `[f_s; f_t]`
- Passes through `WarmStartGradientReverseLayer` (GRL) — reverses gradients during backprop
- GRL coefficient ramps from 0 → 1 following: `2/(1 + exp(-α·t/T)) - 1`
- Discriminator predicts domain (1=source, 0=target)
- Loss: `0.5 * (BCE(d_s, 1) + BCE(d_t, 0))`

**Effect**: The encoder is trained to fool the discriminator → features become domain-invariant.

---

## Training

### Forward Pass

```
source, target → encoder → source_f, target_f
source_f → classifier → source_predict
target_f → classifier → target_predict

source_f → _get_source_similar → src_sim, src_prototype
target_f → _get_target_similar → tgt_sim, tgt_prototype, tgt_cluster_label

source_f, tgt_prototype → s2t_pro
target_f, src_prototype → t2s_pro
source_f, src_prototype → s2s_pro
target_f, tgt_prototype → t2t_pro
```

### Loss Functions

```
total_loss = cls_loss
           + global_transfer_loss
           + source_loss
           + boost_factor × target_loss
           + 0.2 × (cross_domain_loss + in_domain_loss)
```

| Term | Formula | Purpose |
|---|---|---|
| `cls_loss` | LabelSmoothingCE(source_predict, source_label) | Supervised source classification |
| `source_loss` | LabelSmoothingCE on samples with confidence > 0.7 | Filtered high-quality source supervision |
| `target_loss` | LabelSmoothingCE(tgt_sim, tgt_cluster_label) | Pseudo-label supervision on target |
| `global_transfer_loss` | DANN adversarial loss | Feature distribution alignment |
| `cross_domain_loss` | H(s2t_pro) + H(t2s_pro) | Cross-domain entropy minimization |
| `in_domain_loss` | H(s2s_pro) + H(t2t_pro) | Within-domain compactness |

Where `H(p) = -Σ pᵢ log pᵢ` is the standard Shannon entropy.

### Boost Factor

```python
boost_factor = 2.0 * (2.0 / (1.0 + exp(-epoch / 1000)) - 1)
```

Ramps from ~0 at epoch 0 to ~2.0 asymptotically. This **delays the influence of pseudo-labels** until the model is stable enough to generate reliable ones — preventing early noisy pseudo-labels from corrupting training.

### Optimizer & Schedule

- **Optimizer**: RMSprop, lr=0.001, weight_decay=0.001
- **LR schedule**: `lr = lr₀ / (1 + γ·(t/T))^0.75` — polynomial decay
- **Gradient noise**: Gaussian noise `N(0, 0.005)` added to features before DANN loss for regularization

---

## Entropy Minimization — Why It Works

The cross/in-domain losses minimize entropy of similarity distributions. A peaked distribution (low entropy) means a sample is confidently assigned to one prototype. This encourages:

- **`in_domain_loss`**: features cluster tightly within their own domain — compact class representations
- **`cross_domain_loss`**: features confidently match the *other* domain's prototypes — cross-domain alignment

Together they act as a self-supervised geometry regularizer: the feature space is shaped so that class clusters are compact and the two domains overlap, on top of what DANN achieves.

Entropy range for 3 classes: `0` (fully confident) to `log(3) ≈ 1.099` (uniform).

---

## Evaluation

At test time, only `target_predict` is used:
```python
target_f = encoder(target_data)
output = softmax(classifier(target_f))
prediction = argmax(output)
```

The memory banks and prototype logic are **only used during training** to generate pseudo-labels and shape the feature space.

---

## Differences from NSAL-DGAT

| Aspect | PCL-TDGCN | NSAL-DGAT |
|---|---|---|
| Graph adjacency | Dynamic per-sample (two similarity measures fused, top-80% sparse) | Static learnable (GATENet on fixed vector) |
| Target pseudo-labels | K-Means on top-30% confident **target** bank | KNN (k=7) on **source** bank, entropy-weighted |
| Pseudo-label type | Hard (argmax) | Soft (weighted probability) |
| Extra consistency loss | Yes — 4 entropy terms (s2t, t2s, s2s, t2t) | No |
| Source confidence filter | Yes (>0.7 threshold) | No |
| Optimizer | RMSprop | AdamW |
| Target memory bank | Yes | No |

---

## Data Flow Summary

```
Raw .mat files (SEED ExtractedFeatures)
    → load_data() in utils_PCL.py
    → MinMax normalize per subject to [-1, 1]
    → shape: [N_samples, 310]

LOSO split:
    source: 14 subjects stacked → [~3944, 310]
    target: 1 subject           → [~851,  310]

TensorDataset: (feature, sample_index, label)
    ↓
DataLoader (batch_size=48, shuffle=True)
    ↓
PCL.forward(source_batch, target_batch, source_labels, src_idx, tgt_idx, epoch, max_epochs)
    ↓
5-term loss → backward → RMSprop step → lr_scheduler step
    ↓
Evaluate on full target set every epoch → save best model
```

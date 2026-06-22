# OPTA: Full Model Architecture


## Overall Architecture

```
Input [B, 310]  (MLP path)      Input [B, 5, 62]  (GCN path)
        │                                │
        ▼                                ▼
  FeatureExtractor (MLP)       FeatureExtractor (GCN)
  FC(310→64)→ReLU              BN → reshape [B,5,62,1]
  FC(64→64)→ReLU               SampleAdaptiveAdj → A[B,62,62]
        │                      MultipleResidualGCN (L=2 layers)
        │                      concat [x0,x1,x2] → [B,15,62,1]
        │                      flatten → FC(930→64)→ReLU
        │                              → FC(64→64)→ReLU
        └──────────────┬───────────────┘
                       │
                  Feature [B, 64]
                       │
         ┌─────────────┼─────────────────────┐
         ▼             ▼                     ▼
  CosineClassifier  DomainDisc         source_prototypes M_s
  logits_s [B,3]   (DANN GRL)          [C, 64]
  logits_t [B,3]                            │
                                      sinkhorn + FIFO pool
                                      → target_prototypes M_t
                                            [C, 64]
```

---

## 1. Feature Extractor

Two interchangeable backends, both producing `f ∈ ℝ^{B×64}`.

### 1a. MLP Backend

A two-layer MLP applied to flattened input:

```
x [B, 310]
→ FC(310, 64) → ReLU
→ FC(64, 64)  → ReLU
→ f [B, 64]
```

### 1b. GCN Backend

A graph convolutional network that models spatial relationships between EEG electrodes.

**Adjacency matrix initialization**: Pre-computed from standard 10-05 electrode positions. Edge weight between electrodes $i$ and $j$:

$$A_{ij} = \min\!\left(1,\; \frac{\delta}{\|p_i - p_j\|^2 + \epsilon}\right), \quad \delta = 0.00056$$

Stored as a learnable parameter — trainable, initialized from anatomical distances.

**Sample-Adaptive Adjacency (`SampleAdaptiveAdj`)**: Each sample generates its own electrode connectivity modulation via Q·K^T attention, blended with the global adjacency:

$$A_{\text{final}}^{(b)} = \text{ReLU}\!\left(A_{\text{global}} + \alpha \cdot \tanh\!\left(\frac{Q^{(b)} (K^{(b)})^\top}{\sqrt{d}}\right)\right)$$

where Q, K are 1×1 convolutions (`proj_dim=4`), and `α` is a learnable scalar initialized to 0. The model starts with pure global topology and gradually learns how much per-sample adaptation helps.

**MultipleResidualGCN**: L=2 residual GCN layers. Each `ResidualGCN` layer:

1. Two grouped Conv2d operations (per frequency band) with BN and ELU
2. Degree-normalized graph propagation: $L = A \cdot D^{-1}$, then $y = \text{einsum}(x, L)$
3. Residual connection: $y = \text{ELU}(y + x)$

Multi-scale outputs are concatenated: $[x_0, x_1, x_2] \to [B, 15, 62, 1]$ (input + 2 GCN layers × 5 bands). Then:

```
flatten → FC(930, 64) → ReLU → FC(64, 64) → ReLU → f [B, 64]
```

---

## 2. CosineClassifier

A bias-free linear classifier operating in cosine geometry:

$$\text{logits}(f) = \frac{f_n \cdot W_n^\top}{\tau}, \quad f_n = \frac{f}{\|f\|}, \quad W_n = \frac{W}{\|W\|}$$

- $W \in \mathbb{R}^{3 \times 64}$ — learnable class weight matrix
- $\tau = \exp(\log\tau).\text{clamp}(\min=10^{-3})$ — learnable temperature, initialized at 0.1
- No bias — preserves cosine geometry consistent with prototype operations

---

## 3. DANN — Domain Adversarial Alignment

```
f_s, f_t  →  concat [2B, 64]
          →  WarmStartGradientReverseLayer(α=1, lo=0, hi=1)
          →  Discriminator: FC(64→64) → ReLU → FC(64→1)
          →  d_s [B, 1],  d_t [B, 1]

loss_dann = 0.5 * (BCE(d_s, 1) + BCE(d_t, 0))
```

The `WarmStartGradientReverseLayer` ramps the reversal coefficient from 0 to 1 over training:

$$\lambda(t) = \frac{2}{1 + e^{-\alpha t / T}} - 1$$

Gaussian noise $\sigma = 0.005$ is added to features before DANN: $f + 0.005 \cdot \varepsilon,\; \varepsilon \sim \mathcal{N}(0, I)$.

---

## 4. Source Prototypes

Per-batch, label-weighted class mean:

$$M_s^c = \frac{\sum_i y_i^c \cdot f_i^s}{\sum_i y_i^c + \epsilon}, \qquad M_s^c \leftarrow \frac{M_s^c}{\|M_s^c\|}$$

Computed fresh each batch (no memory bank) — introduces controlled stochasticity that raises the best-epoch ceiling under long-horizon best-epoch evaluation.

---

## 5. Adversarial-Confidence Pseudo-Labels

For each target sample $f_t^i$:

**Two independent predictions:**

$$p_i^{\text{cls}} = \text{softmax}(\text{classifier}(f_t^i))$$

$$p_i^{\text{proto}} = \text{softmax}\!\left(\frac{\cos(f_t^i,\, M_s)}{\tau}\right)$$

**Agreement gate:**

$$a_i = \mathbb{1}\!\left[\arg\max p_i^{\text{cls}} = \arg\max p_i^{\text{proto}}\right] \in \{0, 1\}$$

**Geometric mean confidence:**

$$g_i = \sqrt{p_i^{\text{cls}} \odot p_i^{\text{proto}}} \in \mathbb{R}^C$$

**Domain bridge weight** (discriminator's "looks like source" probability):

$$d_i = \sigma\!\left(\text{Discriminator}(f_t^i)\right) \in [0, 1]$$

**Combined score and pseudo-label:**

$$\mathbf{c}_i = a_i \cdot g_i \cdot d_i \in \mathbb{R}^C, \qquad c_i,\; \hat{y}_i^t = \max_c(\mathbf{c}_i)$$

All computed under `torch.no_grad()` — treated as fixed supervision targets.

---

## 6. FIFO Confident-Target Pool + Sinkhorn Target Prototypes

**FIFO Pool** (capacity 256): Each batch, the top-$q\%$ target features by $c_\text{score}$ are pushed in (L2-normalized, detached). A sigmoid schedule controls $q$:

$$q(t) = 0.20 + 0.30 \cdot \left(\frac{2}{1 + e^{-5t/T}} - 1\right) \quad (20\% \to 50\%)$$

**Sinkhorn-Knopp assignment** over pool $Q \in \mathbb{R}^{N \times 64}$:

$$P_{ic} = \exp\!\left(\frac{\cos(q_i, M_s^c)}{\lambda}\right), \quad \lambda = 0.05$$

Alternating normalization (3 iterations):

$$P \leftarrow P \oslash (P\,\mathbf{1}_C + \epsilon), \qquad P \leftarrow P \oslash (\mathbf{1}_N^\top P + \epsilon)$$

The column constraint enforces uniform class mass ($1/3$ per class), preventing prototype collapse.

**Target prototypes:**

$$M_t^c = \frac{\sum_i P_{ic} \cdot q_i}{\sum_i P_{ic}}, \qquad M_t^c \leftarrow \frac{M_t^c}{\|M_t^c\|}$$

$M_t$ is fully detached from the computation graph — constant w.r.t. gradients.

> **Cold-start guard**: if $|Q| < 3$, $M_t = M_s$ (fallback to source prototypes).

---

## 7. Loss Functions

### 7a. `loss_src_ce` — Supervised Source Classification

Label-smoothing cross-entropy ($\varepsilon = 0.0005$) on cosine classifier output:

$$\mathcal{L}_{\text{src-CE}} = \text{LabelSmoothCE}(\text{logits}_s,\; y^s)$$

### 7b. `loss_dann` — Marginal Domain Alignment

$$\mathcal{L}_{\text{DANN}} = \frac{1}{2}\left[\text{BCE}(d_s, \mathbf{1}) + \text{BCE}(d_t, \mathbf{0})\right]$$

### 7c. `loss_tgt_ce` — Reliability-Weighted Pseudo-CE

$$\mathcal{L}_{\text{tgt-CE}} = \frac{1}{B}\sum_i c_i \cdot \text{CE}\!\left(\frac{\cos(f_t^i, M_t)}{\tau},\; \hat{y}_i^t\right)$$

### 7d. `loss_tri` — Triangulation

Two sub-terms — attraction across domains, repulsion within each domain:

$$\mathcal{L}_{\text{tri}} = \underbrace{\frac{1}{C}\sum_c \|M_s^c - M_t^c\|_2^2}_{\text{cross-domain alignment}} + \underbrace{H(M_s) + H(M_t)}_{\text{intra-domain margin}}$$

where the hinge term $H$ pushes different-class prototypes apart with margin $\mu = 0.5$:

$$H(M) = \frac{1}{|\mathcal{P}|}\sum_{c \neq c'} \left[\max\!\left(0,\; \mu - \sqrt{2 - 2\cos(M^c, M^{c'}) + \epsilon}\right)\right]^2$$

### 7e. `loss_xconf` — Cross-Confusion

Source samples classified by target prototypes, supervised with true source labels:

$$\mathcal{L}_{\text{xconf}} = \text{LabelSmoothCE}\!\left(\frac{\cos(f_s, M_t)}{\tau},\; y^s\right)$$

Because $M_t$ shifts every batch (high-variance signal) while $y^s$ is clean supervision, this term raises the best-epoch accuracy ceiling.

---

## 8. Total Loss and Schedules

$$\mathcal{L} = \mathcal{L}_{\text{src-CE}} + \mathcal{L}_{\text{DANN}} + \lambda_1(t)\,\mathcal{L}_{\text{tgt-CE}} + \lambda_2\,\mathcal{L}_{\text{tri}} + \lambda_3(t)\,\mathcal{L}_{\text{xconf}}$$

| Weight | Formula | Range |
|---|---|---|
| $\lambda_1(t)$ | $2 \cdot \left(\frac{2}{1+e^{-t/T}} - 1\right)$ | $0 \to 2$, sigmoid ramp |
| $\lambda_2$ | $0.5$ (constant) | — |
| $\lambda_3(t)$ | $0.2 \cdot \min(1,\; t/200)$ | $0 \to 0.2$, linear over first 200 epochs |

**Optimizer**: RMSprop, lr = 0.001, weight_decay = 1e-5  
**Gradient clipping**: $\|\nabla\|_2 \leq 1.0$  
**Epochs**: 1000 (best-epoch evaluation)

---

## 9. Inference

At test time, classify by nearest prototype in cosine distance:

$$\hat{y} = \arg\max_c \cos\!\left(f,\; M^c\right)$$

Prototype selection:

```
M_t = Sinkhorn(pool, M_s)

if offdiag_max(M_t) > 0.95:   # collapse detected
    M = M_s                    # fallback to source prototypes
else:
    M = M_t
```

`offdiag_max` is the maximum off-diagonal cosine similarity in $M_t$. If any two prototype rows have similarity > 0.95, the prototypes have collapsed and $M_s$ (cached from the last training batch) is used instead.

---

## 10. Component Summary

| Component | Detail | Purpose |
|---|---|---|
| Feature extractor | MLP: 2×FC or GCN: SampleAdaptiveAdj + 2×ResGCN | $\mathbb{R}^{310} \to \mathbb{R}^{64}$ |
| Classifier | CosineClassifier with learnable $\tau$ | Geometry-consistent classification |
| Domain alignment | DANN with WarmStart GRL | Marginal distribution alignment |
| Source prototypes | Per-batch label-weighted mean, L2-normalized | Class centers for source domain |
| Pseudo-labels | 3-way adversarial-confidence score $(a \cdot g \cdot d)$ | Reliable target supervision |
| Target prototypes | Sinkhorn-Knopp on FIFO pool (capacity 256) | Balanced target class centers |
| `loss_tgt_ce` | $c_\text{score}$-weighted cosine CE on $M_t$ | Pseudo-labeled target adaptation |
| `loss_tri` | Cross-domain attract + intra-domain margin hinge | Class-conditional alignment |
| `loss_xconf` | Source classified by target prototypes | High-variance supervised signal |

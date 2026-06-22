# OPTA — Optimal-Transport Prototype Triangulation with Adversarial-Confidence Pseudo-Labels

A new prototype-based EER model designed from first principles. The feature extractor `g(·): ℝ³¹⁰ → ℝ⁶⁴` is treated as a black box and reused unchanged. The four design questions from `idea.md` (classifier, target pseudo-label generation, prototype construction, prototype-based adaptation) are answered below, each with rationale.

---

## Design assumptions

- **Feature extractor** `g(·): ℝ³¹⁰ → ℝ⁶⁴` is fixed (Saber's GCN, or any other). Not designed here.
- **DANN** stays as the marginal-distribution aligner — orthogonal to prototype logic and known to help. We add prototypes *on top* of it.
- **Best-epoch evaluation over 1000 epochs** (per `CLAUDE.md`) means controlled per-step variance is a feature. Designs that over-stabilize (long EMA, large equally-weighted memory banks) are penalized.
- 3 classes, batch size ~96, ~3944 source / ~851 target on SEED LOSO.

---

## 1. Classifier — cosine linear with learnable temperature

**Decision:** A single linear layer `W ∈ ℝ³ˣ⁶⁴` *without bias*, applied as a cosine classifier:

$$\text{logits}(f) = \frac{1}{\tau} \cdot \frac{f \cdot W^\top}{\|f\|_2 \cdot \|W_{c,:}\|_2}$$

with `τ` a learnable scalar (initialized 0.1).

**Why not PRPL's bilinear `U^T V P^T`?**
- PRPL's `S = U^T V` is a rank-32 64×64 matrix; with only 3 output classes, this provides no extra capacity over a plain linear layer. The bilinear form mainly serves PRPL's batch-recompute trick for `P`.
- Mixing dot-product softmax with cosine prototypes creates a geometric inconsistency.

**Why cosine + learnable τ?**
- Cosine geometry matches the prototype losses below (which all operate on cosine similarity). One consistent metric throughout the model.
- Learnable τ removes a free hyperparameter and lets the model commit to its own sharpness level. Standard in modern UDA/FSL.
- One extra scalar parameter — keeps it simple.

**No bias** — bias breaks cosine geometry and obscures prototype interpretability.

---

## 2. Pseudo-label generation — Adversarial-Confidence scoring

### Problem with existing approaches

| Model | Confidence signal | Weakness |
|---|---|---|
| Basic / PCL | `max(softmax(classifier))` | Tautologically confident on source-style features; miscalibrated on target |
| PRPL | Pairwise threshold on cosine sim | No per-sample reliability |
| ADANN | Source-prototype confidence | Single-view; biased toward source |

### Proposal: 3-way agreement weighted by domain bridge

A target sample is reliable iff **two independent predictors agree, weighted by how much the domain discriminator thinks it looks like source.**

For each target sample `f_iᵗ`, compute:
- `pᵢ_cls = softmax(linear classifier(f_iᵗ))`
- `pᵢ_proto = softmax((1/τ) · cos(f_iᵗ, Mˢ))` — similarity to current source prototypes
- `dᵢ ∈ [0,1]` — DANN discriminator's "this looks like source" probability (we already train it; use it as a weight)

**Confidence score (geometric mean, gated by view agreement):**

$$c_i = \mathbb{1}[\arg\max p_i^{\text{cls}} = \arg\max p_i^{\text{proto}}] \cdot \sqrt{p_i^{\text{cls}} \odot p_i^{\text{proto}}} \cdot d_i$$

Take the max-class entry of `cᵢ` as the per-sample reliability scalar. **Pseudo-label** = argmax of the geometric-mean distribution.

**Selection:** top **k%** by `cᵢ` (sigmoid-anneal from 20% → 50%) instead of a fixed threshold γ — quantile-based selection adapts to the domain's confidence scale and never starves.

**Why this works under the variance insight:**
- The classifier and prototype views become decorrelated as training progresses (different inductive biases) → their agreement is a stronger, *stochastic* signal than either alone.
- `dᵢ` is the GRL discriminator we already train. Reusing it as a *confidence weight* (not just a loss) is free.
- Geometric mean penalizes disagreement more sharply than arithmetic — one weak view vetoes the other.

---

## 3. Prototype construction — Sinkhorn-balanced batch prototypes

### 3a. Source prototypes — per-batch label-weighted mean

Don't use a memory bank for source. Compute fresh each batch:

$$M^s_c = \frac{\sum_i \mathbb{1}[y_i = c] \cdot f_i^s}{\sum_i \mathbb{1}[y_i = c] + \epsilon}$$

L2-normalize rows.

**Why batch-wise?** Per the project insight, per-batch prototypes are noisier than memory-bank means → higher per-step variance → higher best-epoch ceiling. Source labels are reliable, so the noise is only sampling noise — useful regularization.

### 3b. Target prototypes — Sinkhorn-Knopp on a confident FIFO pool

Don't use K-Means (no class-balance prior, label-cluster matching is ad-hoc, requires `n_init` randomness). Don't use the classifier's argmax (collapses).

**Steps:**
1. Maintain a **confident-target FIFO pool** `Q` of capacity `≈ 0.3 × |target|` (e.g. 256). Each batch, push the top-k% target features (by `cᵢ` from §2) into `Q`.
2. Build cost matrix from cosine similarity to source prototypes:
   $$L_{ic} = \exp\!\left( \cos(q_i, M^s_c) / \lambda \right)$$
3. **3 iterations of Sinkhorn-Knopp** with row marginal `1/|Q|` and column marginal `1/3` (uniform class prior — SEED is class-balanced) → soft assignment matrix `Π ∈ ℝ^{|Q|×3}`.
4. Target prototypes:
   $$M^t_c = \frac{\sum_i \Pi_{ic} \cdot q_i}{\sum_i \Pi_{ic}}$$
5. L2-normalize rows.

Detach `Π` from the graph for the first 100 epochs (stability), then optionally let gradients flow through (SwAV-style).

**Why Sinkhorn-Knopp instead of K-Means?**

| | K-Means (PCL) | Sinkhorn-Knopp (OPTA) |
|---|---|---|
| Class balance | None — can collapse | Built-in column marginal `1/3` |
| Label-prototype matching | Ad-hoc (`cluster_label`) | Inherited from source via cost matrix |
| Differentiable | No | Yes (iterated row/column normalization) |
| Hyperparameters | `n_init`, `n_clusters` | One temperature `λ` |
| Cost | O(iter · n · k · d) | O(3 · n · 3) per batch — cheap |

Class-balance prior fixes the dominant failure mode of unsupervised target prototypes on SEED: when one emotion has fewer high-confidence target samples, K-Means produces a degenerate 2-cluster solution. Sinkhorn enforces 3 balanced clusters by construction.

**Why FIFO pool, not a full bank?** A full bank averaged uniformly over-stabilizes. A 256-sample rolling pool is smoother than a single batch but still meaningfully shifts each epoch.

---

## 4. Adaptation — Triangulation + Pseudo-CE + Cross-Confusion

Three losses on top of source CE and DANN.

### 4a. Pseudo-CE on target (the workhorse)

$$\mathcal{L}_{\text{tgt-CE}} = -\sum_i c_i \cdot \log p_i^{\text{proto}}[\hat{y}_i^t]$$

Weighted by reliability score `cᵢ` (continuous), not just thresholded. Pseudo-label `ŷᵢᵗ` = argmax of geometric-mean distribution.

### 4b. Class-conditional Triangulation — the new key term

CORAL/MMD/DANN align *marginal* distributions. None aligns the third moment that matters: each class center should land at the same place in source and target.

$$\mathcal{L}_{\text{tri}} = \underbrace{\sum_c \|M^s_c - M^t_c\|_2^2}_{\text{class-anchor alignment}} + \beta \cdot \underbrace{\sum_{c \neq c'} \left[\max(0, \mu - \|M^s_c - M^s_{c'}\|_2)^2 + \max(0, \mu - \|M^t_c - M^t_{c'}\|_2)^2\right]}_{\text{inter-class margin (hinge)}}$$

The first term pulls same-class centers together across domains. The second — borrowed from triplet/contrastive metric learning — pushes different-class centers apart with margin `μ`. Together they **triangulate**: same-class across domains attract, different-class within each domain repel.

**vs ADANN's `loss_cond`:** ADANN uses CE on the prototype similarity matrix with diagonal-as-truth — pushes diagonal up but doesn't enforce inter-class margin. Triangulation does both.

### 4c. Cross-Confusion regularizer — the variance lever

For each labeled source sample `f_iˢ`, compute its similarity to **target** prototypes and supervise with the source's true label:

$$\mathcal{L}_{\text{xconf}} = -\frac{1}{B}\sum_i \log \frac{\exp(\cos(f_i^s, M^t_{y_i}) / \tau)}{\sum_c \exp(\cos(f_i^s, M^t_c) / \tau)}$$

In English: "a labeled source sample should be classifiable by *target* prototypes."

Symmetric in spirit to PCL's `s2t` entropy term but **uses ground-truth labels** instead of minimizing entropy. Because `Mᵗ` shifts every batch (Sinkhorn over a small rolling pool → high variance), this is a high-variance signal *with* a clean supervision label — exactly the regime that should raise the best-epoch ceiling per the project insight.

### 4d. Total loss

$$\mathcal{L} = \mathcal{L}_{\text{src-CE}} + \mathcal{L}_{\text{DANN}} + \lambda_1(t) \cdot \mathcal{L}_{\text{tgt-CE}} + \lambda_2 \cdot \mathcal{L}_{\text{tri}} + \lambda_3(t) \cdot \mathcal{L}_{\text{xconf}}$$

| Term | Weight | Notes |
|---|---|---|
| `src-CE` | 1.0 | Standard label-smoothing CE on labeled source |
| `DANN` | 1.0 | Standard GRL discriminator loss |
| `tgt-CE` | `λ₁(t) = 2·(2/(1+e^(-t/T)) - 1)` | Sigmoid ramp 0 → 2 (same as PCL) |
| `tri` | `λ₂ = 0.5` | Constant; `β = 1.0`, margin `μ = 0.5` |
| `xconf` | `λ₃(t) = 0.2 · min(1, t/200)` | Linear ramp over first 200 epochs — target prototypes too noisy early |

**Deliberate omissions:**
- **No entropy minimization** (PCL's 4 H(·) terms) — collapses distributions, conflicts with the variance insight, and is partially redundant with `tri` + `tgt-CE`.
- **No `‖P^T P − I‖_F` orthogonality** — only meaningful for PRPL's bilinear `P`, not plain prototypes.

---

## Inference

$$\hat{y} = \arg\max_c \cos(g(x),\ M^t_c)$$

using target prototypes from the final epoch's confident pool.

**Collapse fallback** (from ADANN): if any off-diagonal pair of `Mᵗ` has cosine similarity > 0.95, fall back to source prototypes:

$$\hat{y} = \arg\max_c \cos(g(x),\ M^s_c)$$

---

## Why this should beat current SABER on SEED-LOSO

| Component | What it does | Why it helps best-acc ceiling |
|---|---|---|
| Cosine classifier + learnable τ | Geometry-consistent with prototypes | Removes a known classifier-prototype mismatch |
| Adversarial-confidence pseudo-labels | Higher-quality target labels via 3-way agreement | Cleaner training signal |
| Sinkhorn-balanced target prototypes | Class-balanced, source-anchored | Prevents prototype collapse — biggest UDA failure mode |
| FIFO pool (not full bank) | Small, rolling | Keeps useful per-step variance |
| Triangulation loss | Class-anchor alignment + inter-class margin | Adds 3rd-moment alignment DANN/CORAL miss |
| Cross-confusion loss | Source samples classified by target prototypes | High-variance supervised signal — the ceiling-raiser |
| No entropy collapse terms | Removed | Avoids over-stabilizing |

---

## Open questions (test before finalizing)

1. **Detach Sinkhorn assignments?** On-graph could destabilize early training. Default: detached for first 100 epochs, ablate later.
2. **Cross-confusion ramp duration?** 200 epochs is a guess. Could be 100 or 500. Test on one LOSO fold.
3. **Pool capacity?** 256 is a guess (≈ 0.3 × target size). Could be 128 or 512.
4. **Top-k% schedule for confident pool?** 20% → 50% sigmoid is one option; could be linear or constant.

---

## Verifiable implementation roadmap

1. **Skeleton.** `model/opta.py` with cosine classifier + per-batch source prototypes + vanilla pseudo-CE (no Sinkhorn, no triangulation, no cross-confusion). Verify: forward runs, shapes match, 1 CPU epoch finishes.
2. **Add Sinkhorn target prototypes (§3b).** Verify: prototypes don't collapse (off-diagonal cos sim < 0.9 at epoch 100).
3. **Add triangulation (§4b).** Verify: source-target same-class cos sim increases over epochs.
4. **Add adversarial-confidence (§2) + cross-confusion (§4c).** Verify: best-acc ceiling exceeds current SABER on at least 3 LOSO folds.

Each step is a separate commit, each with its own ablation result.

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

Two interchangeable backends, both producing $f \in \mathbb{R}^{B \times 64}$.

### 1a. MLP Backend

EEG signals from multiple electrodes and frequency bands are commonly treated as flattened vectors when spatial modelling is not the primary concern. We provide a lightweight two-layer MLP as a baseline feature extractor:

$$x \xrightarrow{\text{FC}(310,64)} \text{ReLU} \xrightarrow{\text{FC}(64,64)} \text{ReLU} \longrightarrow f \in \mathbb{R}^{64}$$

### 1b. GCN Backend

EEG electrodes are spatially distributed across the scalp according to standardized montages, and functional connectivity between brain regions is known to carry emotion-relevant information. A plain MLP treats all electrode-band combinations as independent features, discarding this spatial structure. We therefore adopt a graph convolutional backbone that explicitly encodes inter-electrode relationships.

**Adjacency matrix initialization.** The initial graph topology is derived from the physical distances between electrodes in the standard 10-05 coordinate system. The edge weight between electrodes $i$ and $j$ is defined as an inverse-square function of their Euclidean separation:

$$A_{\text{global},ij} = \min\!\left(1,\; \frac{\delta}{\|p_i - p_j\|^2 + \epsilon}\right)$$

where $p_i, p_j \in \mathbb{R}^3$ are the 3-D positions of electrodes $i$ and $j$, and $\delta = 0.00056$ is a calibration constant. This initialization encodes the prior that spatially proximate electrodes tend to be more functionally correlated. The adjacency matrix is stored as a learnable parameter and refined jointly with the rest of the network.

**Sample-Adaptive Adjacency.** A shared global adjacency imposes the same connectivity pattern on all samples, ignoring trial-specific neural dynamics. To capture subject- and trial-level variability in functional connectivity, we augment the global graph with a sample-adaptive perturbation computed via scaled dot-product attention:

$$A_{\text{sample}}^{(b)} = \tanh\!\left(\frac{\varphi\!\left((X^{(b)})^\top\right) \cdot \psi(X^{(b)})}{\sqrt{d}}\right)$$

$$A_{\text{final}}^{(b)} = \text{ReLU}\!\left(A_{\text{global}} + \alpha \cdot A_{\text{sample}}^{(b)}\right)$$

where $\varphi$ and $\psi$ are 1d convolutions along electrode and frequency band respectively and $\alpha$ is a learnable scalar initialized to 0. Initializing $\alpha = 0$ ensures that training begins from the anatomically grounded global topology, with sample-specific deviations learned progressively as the model matures.

**Multi-scale Residual GCN.** The graph convolution propagates features across the electrode graph with degree normalization to prevent scale distortion from high-degree nodes. Each residual GCN layer applies:

$$z_m = \delta_2\!\left(D^{-1} A_{\text{final}}^{(b)}\; \delta_1(z_{m-1} W_{m1})\; W_{m2} + z_{m-1}\right)$$

where $D$ is the degree matrix of $A_{\text{final}}^{(b)}$, $\delta_1, \delta_2$ are ELU activations, and $W_{m1}, W_{m2}$ are learnable weight matrices. The residual connection stabilizes gradient flow and prevents over-smoothing of features across layers. To preserve complementary information captured at different propagation depths, outputs from all layers are concatenated:

$$f(x) = \text{concat}(x,\; z_1,\; z_2) \in \mathbb{R}^{15 \times 62}$$

This multi-scale representation is then flattened and projected to the shared 64-dimensional embedding space via two fully-connected layers.

---

## 2. Cosine Classifier

Standard linear classifiers compute raw dot products between features and weight vectors, making their outputs sensitive to feature magnitude. This creates a geometric inconsistency when the same feature space is used for prototype-based losses, which operate on cosine similarity. To ensure a unified metric throughout the model, we adopt a bias-free cosine classifier with a learnable temperature:

$$\text{logits}(f) = \frac{f_n \cdot W_n^\top}{\tau}, \quad f_n = \frac{f}{\|f\|}, \quad W_n = \frac{W}{\|W\|}$$

- $W \in \mathbb{R}^{3 \times 64}$ — learnable class weight matrix, each row representing a class direction on the unit hypersphere
- $\tau = \exp(\log\tau).\text{clamp}(\min=10^{-3})$ — learnable temperature initialized at 0.1, controlling the sharpness of the output distribution
- No bias term — a bias would shift the decision boundary away from the origin, breaking consistency with the cosine-based prototype distances

The temperature $\tau$ is parameterized as $\exp(\log\tau)$ to ensure strict positivity, and the model learns the appropriate confidence calibration directly from data rather than relying on manual tuning.

---

## 3. DANN — Domain Adversarial Alignment

A fundamental challenge in cross-subject EEG recognition is the distributional shift between subjects: features extracted from source subjects occupy a different region of the feature space than those from the target subject, even for the same emotional state. We employ Domain-Adversarial Neural Network (DANN) training to align the marginal feature distributions of the two domains.

A domain discriminator $D(\cdot)$ is trained to distinguish source from target features, while the feature extractor is simultaneously trained — via a Gradient Reversal Layer (GRL) — to produce representations that are indistinguishable across domains:

$$\mathcal{L}_{\text{DANN}} = \frac{1}{2}\left[\text{BCE}(D(f_s), \mathbf{1}) + \text{BCE}(D(f_t), \mathbf{0})\right]$$

The GRL coefficient follows a warm-start sigmoid schedule, ramping from 0 to 1 over training:

$$\lambda(t) = \frac{2}{1 + e^{-\alpha t / T}} - 1$$

This schedule prevents the adversarial signal from destabilizing early feature learning before the classifier has converged to a reasonable solution. Small Gaussian perturbations ($\sigma = 0.005$) are added to features before the discriminator to improve training robustness.

---

## 4. Source Prototypes

Class prototype vectors serve as anchors for both pseudo-label generation and domain alignment. Following PRPL, source prototypes are computed as the label-weighted mean of features within each training batch:

$$M_s^c = \frac{\sum_i y_i^c \cdot f_i^s}{\sum_i y_i^c + \epsilon}, \qquad M_s^c \leftarrow \frac{M_s^c}{\|M_s^c\|}$$

where $y_i^c \in \{0,1\}$ indicates whether source sample $i$ belongs to class $c$. Per-batch computation means the prototypes reflect the current mini-batch. Prototype-based adaptation benefits from prototypes that evolve across batches rather than converging to a fixed point, as gradual drift encourages the feature extractor to explore a broader region of the parameter space and discover better-performing configurations during training.

---

## 5. Adversarial-Confidence Pseudo-Labels

Assigning pseudo-labels to unlabeled target samples is the central challenge in unsupervised domain adaptation. A naive approach — taking the argmax of the classifier's softmax output — is unreliable because the classifier is trained solely on source data and may be miscalibrated on target samples. To obtain more trustworthy pseudo-labels, we propose an adversarial-confidence scoring mechanism that aggregates three complementary reliability signals.

**Two independent predictions.** For each target sample $f_t^i$, two independent class probability estimates are obtained from predictors with different inductive biases:

$$p_i^{\text{cls}} = \text{softmax}(\text{classifier}(f_t^i))$$

$$p_i^{\text{proto}} = \text{softmax}\!\left(\frac{\cos(f_t^i,\, M_s)}{\tau}\right)$$

The first is produced by the learned cosine classifier; the second is derived from direct geometric proximity to source class prototypes. Because these two estimators are structurally independent, their agreement provides a stronger reliability signal than either alone.

**Agreement gate.** A binary gate filters out samples where the two predictors disagree on the predicted class, since such samples are inherently ambiguous:

$$a_i = \mathbb{1}\!\left[\arg\max p_i^{\text{cls}} = \arg\max p_i^{\text{proto}}\right] \in \{0, 1\}$$

**Geometric mean confidence.** For samples that pass the agreement gate, the per-class confidence is measured as the geometric mean of the two probability distributions:

$$g_i = \sqrt{p_i^{\text{cls}} \odot p_i^{\text{proto}}} \in \mathbb{R}^C$$

The geometric mean penalizes cases where one predictor is confident but the other is not more severely than the arithmetic mean. For example, if $p_i^{\text{cls}} = 0.35$ and $p_i^{\text{proto}} = 0.95$, the geometric mean yields $\sqrt{0.35 \times 0.95} \approx 0.58$, substantially lower than the arithmetic mean of 0.65, reflecting the genuine uncertainty.

**Domain bridge weight.** A target sample that lies in a region of feature space already well-covered by source data is more likely to be reliably classified, since both the classifier and the source prototypes have seen training signal in that neighborhood. We quantify this proximity to the source distribution using the domain discriminator's output — the same network trained adversarially to distinguish source from target:

$$d_i = \sigma\!\left(\text{Discriminator}(f_t^i)\right) \in [0, 1]$$

A high $d_i$ indicates that the feature extractor has successfully mapped this target sample into the source-like region of the feature space, making its pseudo-label more trustworthy.

**Combined score and pseudo-label.** The three signals are fused multiplicatively, implementing an AND-style logic where all three conditions must be satisfied for a high reliability score:

$$\mathbf{c}_i = a_i \cdot g_i \cdot d_i \in \mathbb{R}^C, \qquad c_i,\; \hat{y}_i^t = \max_c(\mathbf{c}_i)$$

The scalar $c_i$ serves as a per-sample reliability weight in the subsequent pseudo-label loss, and $\hat{y}_i^t$ is the assigned pseudo-label. All computations in this block are performed without gradients, as $c_i$ and $\hat{y}_i^t$ are treated as fixed supervision targets.

---

## 6. FIFO Confident-Target Pool + Sinkhorn Target Prototypes

Constructing reliable target-domain prototypes without label supervision is inherently difficult. Two natural approaches both suffer from well-known failure modes.

The first is to cluster target features directly using an algorithm such as K-Means. Clustering imposes no class-balance constraint: in domains where one emotion class happens to produce more high-confidence or spatially concentrated features, the algorithm produces a degenerate solution in which one or two clusters absorb the majority of samples while a third cluster covers almost none. Moreover, the correspondence between discovered clusters and actual emotion classes must be resolved by a separate, heuristic label-assignment step, which introduces additional error.

The second is to assign each target sample to a class using the model's own classifier — taking the argmax of the softmax output as a hard assignment, meaning each sample receives exactly one class label with no uncertainty. Hard assignments based on a classifier trained solely on source data are unreliable on out-of-distribution target samples and, more critically, introduce a self-reinforcing bias: if the classifier initially over-predicts one class on the target domain, the prototypes derived from those assignments will further reinforce that prediction in the next iteration, leading to progressive collapse.

We address both failure modes through a combination of selective memory accumulation and optimal-transport-based prototype estimation, which replaces hard assignments with soft, globally balanced assignments.

**FIFO Confident-Target Pool.** To accumulate a representative and reliable set of target features for prototype estimation, we maintain a fixed-capacity FIFO buffer of size 256. At each training step, the top-$q\%$ target features ranked by reliability score $c_i$ are pushed into the pool (L2-normalized, detached from the computation graph). A sigmoid schedule gradually relaxes the selection criterion as the model improves:

$$q(t) = 0.20 + 0.30 \cdot \left(\frac{2}{1 + e^{-5t/T}} - 1\right) \quad (20\% \to 50\%)$$

Early in training, only the most confidently aligned target samples enter the pool; as domain alignment improves, the acceptance threshold is relaxed to broaden coverage. The rolling FIFO structure ensures that the pool reflects recent model state rather than averaging over the entire training history or only current mini-batch.

**Sinkhorn-Knopp Assignment.** Given the pool $Q \in \mathbb{R}^{N \times 64}$, we seek soft class assignments that are both anchored to the source prototype geometry and balanced across classes. Unconstrained soft assignment tends to collapse onto the most discriminative class; we prevent this by framing the assignment as an optimal transport problem with a uniform marginal constraint, solved via the Sinkhorn-Knopp algorithm.

The assignment matrix is initialized from the cosine similarity between each pool sample and the source prototypes:

$$P_{ic} = \exp\!\left(\frac{\cos(q_i, M_s^c)}{\lambda}\right), \quad \lambda = 0.05$$

Alternating row and column normalizations are then applied for 3 iterations:

$$P \leftarrow P \oslash (P\,\mathbf{1}_C + \epsilon), \qquad P \leftarrow P \oslash (\mathbf{1}_N^\top P + \epsilon)$$

where $\oslash$ denotes element-wise division. Row normalization ensures each sample has a valid probability distribution over classes; column normalization enforces a uniform class marginal of $1/C$, guaranteeing that all three emotion classes receive equal total assignment mass. This constraint is the key mechanism that prevents degenerate solutions where the majority of target samples are assigned to a single dominant class.

**Target prototypes.** The class-conditional target prototype is computed as the soft-assignment-weighted centroid of pool features:

$$M_t^c = \frac{\sum_i P_{ic} \cdot q_i}{\sum_i P_{ic}}, \qquad M_t^c \leftarrow \frac{M_t^c}{\|M_t^c\|}$$

$M_t$ is fully detached from the computation graph and serves as a fixed reference for the downstream losses.

> **Cold-start guard**: if $|Q| < 3$, $M_t = M_s$ (fallback to source prototypes).

---

## 7. Loss Functions

$$\mathcal{L} = \mathcal{L}_{\text{src-CE}} + \mathcal{L}_{\text{DANN}} + \lambda_1(t)\,\mathcal{L}_{\text{tgt-CE}} + \lambda_2\,\mathcal{L}_{\text{tri}} + \lambda_3(t)\,\mathcal{L}_{\text{xconf}}$$

### 7a. Supervised Source Classification

The primary supervised signal is a label-smoothing cross-entropy loss on the source domain, which penalizes overconfident predictions and improves calibration:

$$\mathcal{L}_{\text{src-CE}} = \text{LabelSmoothCE}(\text{logits}_s,\; y^s), \quad \varepsilon = 0.0005$$

### 7b. Domain Adversarial Loss

As described in Section 3, the DANN loss aligns the marginal feature distributions of the two domains:

$$\mathcal{L}_{\text{DANN}} = \frac{1}{2}\left[\text{BCE}(d_s, \mathbf{1}) + \text{BCE}(d_t, \mathbf{0})\right]$$

### 7c. Reliability-Weighted Pseudo-CE

To leverage the unlabeled target data, we apply cross-entropy supervision using the pseudo-labels from Section 5. Rather than applying uniform weight to all pseudo-labeled samples, each sample's contribution is scaled by its reliability score $c_i$, ensuring that noisy pseudo-labels exert proportionally less influence on the gradient:

$$\mathcal{L}_{\text{tgt-CE}} = \frac{1}{B}\sum_i c_i \cdot \text{CE}\!\left(\frac{\cos(f_t^i, M_t)}{\tau},\; \hat{y}_i^t\right)$$

The classification target is the cosine similarity to target prototypes $M_t$ rather than the classifier weights, encouraging target features to cluster around their own domain's class centers.

### 7d. Triangulation Loss

DANN aligns only the marginal feature distributions, without any guarantee that the class-conditional distributions are co-located across domains. A well-aligned representation requires that features of the same class in both domains map to the same region of the feature space. The triangulation loss enforces this class-conditional alignment through two complementary terms:

$$\mathcal{L}_{\text{tri}} = \underbrace{\frac{1}{C}\sum_c \|M_s^c - M_t^c\|_2^2}_{\text{cross-domain attraction}} + \underbrace{H(M_s) + H(M_t)}_{\text{intra-domain separation}}$$

The first term directly minimizes the squared distance between source and target prototype pairs of the same class, pulling same-class centers together across domains. The second term applies a margin-based hinge penalty to prevent different-class prototypes within each domain from collapsing toward one another:

$$H(M) = \frac{1}{|\mathcal{P}|}\sum_{c \neq c'} \left[\max\!\left(0,\; \mu - \sqrt{2 - 2\cos(M^c, M^{c'}) + \epsilon}\right)\right]^2, \quad \mu = 0.5$$

Together, these two terms impose a triangulation geometry: same-class prototypes across domains are attracted, while different-class prototypes within each domain are repelled, yielding a well-structured and discriminative feature space.

### 7e. Cross-Confusion Loss

Even after triangulation, there is no direct training signal that forces the source feature extractor to be compatible with the target prototype geometry. We introduce a cross-confusion loss that addresses this by requiring labeled source samples to be correctly classifiable using the target prototypes:

$$\mathcal{L}_{\text{xconf}} = \text{LabelSmoothCE}\!\left(\frac{\cos(f_s, M_t)}{\tau},\; y^s\right)$$

This loss is complementary to $\mathcal{L}_{\text{tgt-CE}}$: while $\mathcal{L}_{\text{tgt-CE}}$ pushes target features toward target prototype regions, $\mathcal{L}_{\text{xconf}}$ simultaneously constrains source features to lie within the same regions. The coupling between clean source supervision and the dynamically updated target prototypes creates a mutually consistent feature space in which both domains share a common class geometry.

---

## 8. Training Schedule

$$\mathcal{L} = \mathcal{L}_{\text{src-CE}} + \mathcal{L}_{\text{DANN}} + \lambda_1(t)\,\mathcal{L}_{\text{tgt-CE}} + \lambda_2\,\mathcal{L}_{\text{tri}} + \lambda_3(t)\,\mathcal{L}_{\text{xconf}}$$

The loss weights are scheduled to reflect the reliability of each signal at different stages of training:

| Weight | Formula | Range | Rationale |
|---|---|---|---|
| $\lambda_1(t)$ | $2 \cdot \left(\frac{2}{1+e^{-t/T}} - 1\right)$ | $0 \to 2$, sigmoid | Pseudo-labels are unreliable early; the sigmoid ramp delays their influence until the feature extractor has stabilized |
| $\lambda_2$ | $0.5$ (constant) | — | Prototype alignment is beneficial throughout training |
| $\lambda_3(t)$ | $0.2 \cdot \min(1,\; t/200)$ | $0 \to 0.2$, linear over 200 epochs | Target prototypes are poorly initialized in early training; the linear ramp prevents the cross-confusion loss from providing misleading gradients before the pool accumulates sufficient reliable samples |

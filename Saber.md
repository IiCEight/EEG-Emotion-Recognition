# SABER Model Architecture Overview

This document summarizes the architecture and training workflow implemented by the current codebase, mainly from `model/saber.py` and `train/training.py`.

## 1. High-Level Idea

SABER is a **domain-adversarial EEG emotion recognition model** with two coupled objectives:

- **Task objective**: classify source-domain EEG samples into emotion classes.
- **Domain objective**: make extracted features domain-invariant by confusing a domain discriminator through gradient reversal.

The model follows a DANN-style pattern:

1. Shared feature extractor processes both source and target inputs.
2. Classifier predicts emotion labels for source features.
3. Domain discriminator predicts source vs. target domain for both feature sets.
4. Gradient reversal in front of the discriminator forces the feature extractor to remove domain-specific cues.

## 2. End-to-End Data Flow

### Input formatting in training

In training (`train/training.py`), input data is rearranged from:

- `sample x chan x feature`

to:

- `sample x feature x chan`

using `einops.rearrange`.

Inside `FeatureExtractor.forward`, tensors are explicitly reshaped to:

- `B x 5 x 62` then unsqueezed to `B x 5 x 1 x 62`

so the implementation currently assumes **5 frequency-band features** and **62 electrodes**.

### Dual-domain forward pass

`Saber.forward(source, target)`:

1. `source_f = feature_extractor(source)`
2. `target_f = feature_extractor(target)`
3. `class_output = class_classifier(source_f)`
4. `domain_output_source = domain_classifier(GRL(source_f))`
5. `domain_output_target = domain_classifier(GRL(target_f))`

Returns:

- class logits for source samples,
- source-domain logits,
- target-domain logits.

For evaluation, `Saber.predict(x)` runs only feature extractor + class classifier.

## 3. SABER Module Composition

## 3.1 Top-level `Saber`

Main components:

- `FeatureExtractor`
- `Classifier` (linear head for emotion classes)
- `Discriminator` (2-class domain head)
- `WarmStartGradientReverseLayer` (GRL)

Configured constants in current code:

- `hidden_1 = 256` (declared, not used directly in the final MLP output path)
- `hidden_2 = 64` (feature embedding dimension to both heads)
- `grad_reverse_max_iter = 1000`

## 3.2 FeatureExtractor

The feature extractor is graph-based and attention-enhanced.

### Learnable graph priors

- Two learnable adjacency parameters: `adj_a`, `adj_b`.
- Both initialized from `get_adj_from_standard()`.

### Two parallel graph streams

- `MRGCN_a`: `MulipleResidualGCN`
- `MRGCN_b`: `MulipleResidualGCN`

Each stream applies:

1. `RemapAdjacencyMatrix`: fully connected remapping of flattened adjacency matrix.
2. A stack of residual graph-convolution blocks (`layers`, default = 2).
3. Concatenation of intermediate outputs along channel dimension (`layers + 1` feature groups).

Then stream outputs are fused by averaging:

- `g_feat = (g_feat_a + g_feat_b) / 2`
- `g_adj = (g_adj_a + g_adj_b) / 2` (computed but not used downstream).

### Attention block

`CBAMBlock` is applied to fused graph features:

- Channel attention (global avg/max pooling + bottleneck conv MLP)
- Spatial attention (conv over concatenated channel max/mean maps)
- Residual addition (`out + residual`)

### Projection head

After attention:

1. Flatten all non-batch dimensions.
2. `fc1`: projects `chan_num * (layers + 1) * band_num` to `hidden_2`.
3. ReLU
4. `fc2`: `hidden_2 -> hidden_2`
5. ReLU

Output is a 64-d feature vector per sample.

(There are dropout layers defined but currently commented out in forward.)

## 3.3 Classifier Head

`Classifier` is a single linear layer:

- `Linear(64, num_classes)`

It outputs class logits used with cross-entropy.

## 3.4 Domain Discriminator Head

`Discriminator` structure:

1. `Linear(64, 64)`
2. ReLU
3. `Linear(64, 2)`

Outputs domain logits for source/target. Dropout and sigmoid are present in module definition but not used in forward.

## 3.5 Gradient Reversal Layer (GRL)

`WarmStartGradientReverseLayer` behavior:

- Forward pass: identity transform.
- Backward pass: multiplies gradients by `-coeff`.
- `coeff` follows a logistic warm-start schedule from `low` to `high` as iteration count grows.
- With `auto_step=True`, internal step counter increments each GRL call.

This mechanism encourages domain confusion while preserving task-discriminative structure.

## 4. Training Objective and Procedure

Implemented in `train/training.py`.

## 4.1 Batching strategy

- Source batches come from training loader (`train_loader`).
- Target batches come from test loader (`test_loader`), wrapped in an infinite iterator (`pytorch_safe_cycle`) to always provide target data.

Domain labels per batch:

- source domain label = 0
- target domain label = 1

## 4.2 Loss function

Let:

- $L_{cls}$ = cross-entropy between source class logits and source emotion labels
- $L_{dom,s}$ = cross-entropy between source domain logits and zeros
- $L_{dom,t}$ = cross-entropy between target domain logits and ones

Code combines losses as:

$$
L = L_{cls} + 0.5\,L_{dom,s} + L_{dom,t}
$$

All losses are optimized jointly with AdamW.

## 4.3 Optimizer and LR schedule

- Optimizer: `AdamW`, `weight_decay = 0.001`
- Learning rate scheduler: `LambdaLR` with

$$
\lambda(e) = \frac{1}{\left(1 + 10\cdot\frac{e}{\max(1, E)}\right)^{0.75}}
$$

where $e$ is epoch and $E$ is total epochs.

A custom `StepwiseLR_GRL` scheduler class exists in the file but is currently not used.

## 5. Evaluation Path

At each epoch end:

1. Model switches to eval mode.
2. `model.predict(test_data)` produces class logits.
3. Argmax gives predicted labels.
4. Accuracy is computed and recorded via `Metric.update(subject_id, session_id, acc)`.

No domain branch is used during inference.

## 6. Shape and Configuration Notes

- Current implementation hardcodes reshape assumptions `5` features and `62` channels in `FeatureExtractor.forward`.
- `train()` rearranges data to `sample x feature x chan`, consistent with the extractor’s expected reshape.
- In `main.py`, the model class is selected via `MODEL[model_name]` and trained per subject/session split.

## 7. Conceptual Summary

SABER can be viewed as:

- **Graph feature learner** (dual MRGCN streams + CBAM attention),
- **Task predictor** (emotion classifier),
- **Adversarial domain aligner** (GRL + domain discriminator).

The combined objective encourages features that remain discriminative for emotion recognition while becoming less sensitive to domain shift between source and target EEG distributions.

# What You Need To Do
I want to improve the Two parallel graph streams. And the intuition is that I want different graph streams to learn different emotions separately. For example, One branch like the emotions about positive, and other learn the negative. Help me implement this idea innovatively and create a good way to fuse these graph streams. Current fusion way is averaging them, but it doesn't work well(No improve).
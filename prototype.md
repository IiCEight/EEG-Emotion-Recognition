# Read this paper

> Read this paper
>
> User context
>
> Attachments:
>
> - EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf: `./DANN/EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf`

<details><summary>6 previous messages</summary>


> I’ll read the PDF and pull out the core contribution, method, experiments, and takeaways. I’ll start by extracting the text so I can give you a useful map of the paper rather than a vague skim.
>
> <details><summary>Ran 3 commands</summary>
>
>
> - Ran `Get-Item -LiteralPath 'D:\MasterSecondStage\paper\DANN\EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf' | Select-Object FullName,Length,LastWriteTime`
> - Ran `Get-Command pdftotext -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source`
> - Ran `Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source`
>
> </details>
>
> The PDF is reachable and `pdftotext` is available, so I can inspect it locally without guessing. Next I’m extracting the text and checking the structure: abstract, method section, experiments, and conclusions.
>
> <details><summary>Ran 2 commands</summary>
>
>
> - Ran `pdfinfo 'D:\MasterSecondStage\paper\DANN\EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf'`
> - Ran `pdftotext -layout -enc UTF-8 'D:\MasterSecondStage\paper\DANN\EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf' -`
>
> </details>
>
> The extracted paper is 5 pages from IEEE BIBM 2025. I’ve got the headline result already; I’m digging into the methodology details now so the summary captures what PAF-CPA actually does, not just the abstract claims.
>
> <details><summary>Ran 3 commands</summary>
>
>
> - Ran `pdftotext -layout -enc UTF-8 'D:\MasterSecondStage\paper\DANN\EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf' - | Select-String -Pattern 'METHODOLOGY|feature extraction|prototype|similarity|loss|EXPERIMENT|DATASET|Experimental|Ablation' -Context 2,6`
> - Ran `pdftotext -raw -enc UTF-8 'D:\MasterSecondStage\paper\DANN\EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf' - | Select-String -Pattern 'where|The total loss|Algorithm|SEED|SEED-IV|DE feature|Adam|batch|epoch|learning rate' -Context 2,4`
> - Ran `pdftotext -layout -enc UTF-8 'D:\MasterSecondStage\paper\DANN\EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf' - | Select-String -Pattern 'TABLE I|TABLE II|TABLE III|TABLE IV|TABLE V|TABLE VI|PAF-CPA|Baseline' -Context 0,12`
>
> </details>
> </details>

I read it. The paper proposes **PAF-CPA**, a domain adaptation framework for cross-subject EEG emotion recognition.

**Main idea:** EEG emotion models struggle because different subjects produce different EEG distributions. PAF-CPA combines three things:

1. **DANN-style adversarial alignment** to learn shared source/target feature space.
2. **Prototype adaptation**: build class prototypes from source samples, select high-confidence target samples, then progressively fuse source and target prototypes.
3. **Fine-grained contrastive losses**: pairwise similarity loss plus class-specific similarity loss, encouraging same-emotion samples to cluster and different emotions to separate.

**Datasets and setup:** It evaluates on **SEED** and **SEED-IV**, using leave-one-subject-out cross-validation. Inputs are DE features from 62-channel EEG segments: 1-second windows for SEED, 4-second windows for SEED-IV.

**Reported results:**

| Protocol                    |    PAF-CPA result |
| --------------------------- | ----------------: |
| SEED single-session LOSO    |  **95.30 ± 4.16** |
| SEED cross-session LOSO     |  **86.11 ± 4.70** |
| SEED-IV single-session LOSO | **78.22 ± 10.59** |
| SEED-IV cross-session LOSO  |  **69.33 ± 8.53** |

It is strongest on SEED, where it beats the listed baselines. On **SEED-IV cross-session**, it is competitive but **not the best**: PR-PL reports 74.92 ± 7.92, higher than PAF-CPA’s 69.33 ± 8.53.

**Ablation takeaway:** The biggest gain comes from adding **source prototypes** to the adversarial baseline. Progressive beta scheduling and the final class-specific contrastive component add smaller but consistent improvements on SEED.

**My read:** The paper’s useful contribution is not just “DANN plus contrastive learning,” but the way it uses **prototype transfer with gradually increasing target influence**. The main limitation is that SEED-IV cross-session performance is weaker than some prior work, and the paper does not deeply analyze why the method transfers less well there.

> Show me the details of Prototype adaptation

<details><summary>4 previous messages</summary>


> I’ll zoom in on the prototype-adaptation section and reconstruct the exact flow: source prototypes, target sample selection, target prototypes, and progressive fusion. The PDF extraction is a bit messy around formulas, so I’m going to pull that area more carefully.
>
> <details><summary>Ran 2 commands</summary>
>
>
> - Ran `pdftotext -raw -enc UTF-8 'D:\MasterSecondStage\paper\DANN\EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf' - | Select-String -Pattern 'Prototype Feature Extraction|For source domain processing|target domain data|cross-domain similarity|screened samples|progressive fusion|Final feature representations|softmax' -Context 4,12`
> - Ran `pdftotext -layout -f 2 -l 2 -enc UTF-8 'D:\MasterSecondStage\paper\DANN\EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf' -`
>
> </details>
>
> I found the core equations on page 2. There’s one notation quirk: the target data is formally written with labels, but the experiment section says only raw target data is used, so those target “emotion categories” are inferred pseudo-groups from similarity/prediction rather than true labels during training.
>
> <details><summary>Ran 2 commands</summary>
>
>
> - Ran `pdftotext -layout -f 3 -l 3 -enc UTF-8 'D:\MasterSecondStage\paper\DANN\EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf' - | Select-Object -First 90`
> - Ran `pdftotext -raw -f 2 -l 3 -enc UTF-8 'D:\MasterSecondStage\paper\DANN\EEG_Emotion_Recognition_via_Prototype_Adaptive_Fine-Grained_Contrastive_Progressive_Alignment.pdf' - | Select-String -Pattern "X't|μ|beta|β|topK|alpha|α|screened|Sμ|softmax|K=" -Context 1,3`
>
> </details>
> </details>

**Prototype Adaptation Details**

PAF-CPA’s prototype adaptation is the part that transfers class structure from the labeled source subjects to the unlabeled target subject.

The flow is:

```text
source labeled features
        ↓
source class prototypes
        ↓
compare target samples to source classes
        ↓
select high-confidence target samples
        ↓
target pseudo-prototypes
        ↓
progressively fuse source + target prototypes
        ↓
prototype-based classification/alignment
```

**1. Source Class Prototypes**

The source data is split by emotion label:

```text
D_s,m = source samples whose label = m
```

For each emotion class `m`, the source prototype is the centroid of the feature vectors in the current mini-batch:

```text
mu_s^m = (1 / |X_s,m|) * sum f(x_s_i), for x_s_i in X_s,m
```

So each prototype is basically the average representation of one emotion class after the feature extractor.

Example:

```text
mu_s^positive
mu_s^neutral
mu_s^negative
```

for SEED.

**2. Target Sample Confidence via Source Similarity**

Target samples are unlabeled, so the method estimates which target samples probably belong to each class.

For each class `m`, it computes the average cosine similarity between a target sample and source samples from class `m`:

```text
S_m = average sim(f(x_s_i), f(x_t_j))
```

where `sim` is cosine similarity:

```text
sim(x_i, x_j) = (x_i · x_j) / (||x_i|| * ||x_j||)
```

Interpretation:

```text
If a target sample is very similar to source positive samples,
it is treated as a high-confidence positive-like target sample.
```

**3. Top-K Target Selection**

For each emotion class, the framework selects the top-`K` most similar target samples:

```text
X'_t,n = topK(S_n, X_t,n)
K = alpha * |X_t,n|
```

`alpha` is the selection ratio. In experiments:

```text
SEED:    alpha in (0.15, 0.25)
SEED-IV: alpha in (0.1, 0.2)
```

So the target prototype is not built from all target samples. It is built only from samples that look confident enough under source-class similarity.

**4. Target Prototypes**

After screening high-confidence target samples, the target prototype for class `n` is computed as:

```text
mu_t^n = (1 / |X'_t,n|) * sum f(x_t_i), for x_t_i in X'_t,n
```

This is like a pseudo-label prototype: the model says, “these target samples probably represent class `n`, so average them.”

**5. Progressive Prototype Fusion**

The final adapted prototype is a weighted fusion of source and target prototypes:

```text
mu_i = (1 - beta) * mu_s^m + beta * mu_t^n
```

`beta` controls how much the model trusts the target-domain prototype.

Early training:

```text
beta small -> rely mostly on source prototypes
```

Later training:

```text
beta larger -> adapt more toward target prototypes
```

The paper uses a stepwise schedule. For SEED, training is divided into 5 segments of 100 epochs; in the first 4 segments, `beta` increases by 0.25 during the first 50 epochs, then stays fixed for the remaining 50 epochs. SEED-IV uses the same idea but with 1500-epoch segments.

**6. Classification With Adapted Prototypes**

Finally, the model compares each feature vector `f(x_i)` with every adapted prototype:

```text
p = softmax(
  f(x_i)^T S_mu1,
  f(x_i)^T S_mu2,
  ...,
  f(x_i)^T S_muc
)
```

So prediction is prototype-similarity based: the class whose adapted prototype is most aligned with the sample gets the highest probability.

**Why This Matters**

The method avoids immediately trusting noisy target pseudo-labels. It starts from reliable source prototypes, selects only confident target samples, then gradually shifts the class anchors toward the target subject. That is the “adaptive progressive alignment” part.

> Can you render the latex formula since now is text

Sure. Here are the prototype-adaptation formulas rendered in LaTeX.

**1. Source Class Prototype**

For source samples of emotion class \(m\):

\[
D_{s,m}=\{(X_{s,m},Y_{s,m})\in D_s \mid Y_{s,m}=m\}, \quad m=1,2,\dots,C
\]

The source prototype is:

\[
\mu_s^m=
\frac{1}{|X_{s,m}|}
\sum_{x_i^s \in X_{s,m}} f(x_i^s)
\]

where \(f(\cdot)\) is the feature extractor.

**2. Cross-Domain Similarity**

For each target sample, compute its similarity to source samples of class \(m\):

\[
S_m=
\frac{1}{|X_{s,m}|}
\sum_{x_i^s \in X_{s,m},\, x_j^t \in X_t}
\operatorname{sim}\left(f(x_i^s), f(x_j^t)\right)
\]

Cosine similarity is:

\[
\operatorname{sim}(x_i,x_j)=
\frac{x_i \cdot x_j}
{\|x_i\|_2 \|x_j\|_2}
\]

**3. High-Confidence Target Sample Selection**

Select the top-\(K\) most similar target samples:

\[
X'_{t,n}=
\left\{
x_k^t \in \operatorname{topK}(S_n, X_{t,n})
\right\}
\]

\[
K=\alpha |X_{t,n}|
\]

where \(\alpha\) is the selection ratio.

**4. Target Prototype**

The target prototype is computed from the selected target samples:

\[
\mu_t^n=
\frac{1}{|X'_{t,n}|}
\sum_{x_i^t \in X'_{t,n}} f(x_i^t)
\]

**5. Progressive Prototype Fusion**

The adapted prototype is a weighted combination of source and target prototypes:

\[
\mu_i =
(1-\beta)\mu_s^m + \beta \mu_t^n
\]

where:

\[
\beta \in (0,1), \quad i=1,2,\dots,C
\]

Early training uses small \(\beta\), relying more on source prototypes. Later training increases \(\beta\), allowing stronger target-domain adaptation.

**6. Prototype-Based Classification**

Class probabilities are computed by comparing the sample feature with each adapted prototype:

\[
p=
\operatorname{softmax}
\left(
f(x_i)^T S_{\mu_1},
f(x_i)^T S_{\mu_2},
\dots,
f(x_i)^T S_{\mu_C}
\right)
\]

In plain terms: each class gets a score based on how close the sample feature is to that class’s adapted prototype.
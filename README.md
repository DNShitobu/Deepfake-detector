---
license: mit
language:
  - en
tags:
  - deepfake-detection
  - multimodal
  - audio-visual
  - cross-modal-attention
  - video-classification
  - digital-forensics
pipeline_tag: video-classification
library_name: transformers
---

# Hybrid Multi-Modal Deepfake Detector

A reference implementation of the detection paradigm identified as the strongest
in the systematic literature review **"Single-Modal versus Hybrid Multi-Modal
Frameworks for the Detection of Deepfake Videos"** (Fuseini, 2026). The review
screened 8,247 records under PRISMA 2020 and synthesised 59 studies. It found
that hybrid multi-modal frameworks, which exploit cross-channel consistency
between audio and video, outperform single-modal detectors on the harder
benchmarks (DFDC, FakeAVCeleb, AV-Deepfake1M).

This repository ships **three interchangeable architectures** behind one Hugging
Face model class, selectable through the config `architecture` field:

| `architecture` | Description | Review paradigm |
|----------------|-------------|-----------------|
| `saff_plus`    | **The strongest architecture (default).** Modality-specific transformer self-attention encoders with learnable CLS tokens, stacked bidirectional cross-modal attention, a Cross-Modal Graph Attention layer (CM-GAN style), an offset-tolerant soft synchronisation module, an auxiliary cross-modal temporal prediction task, a frame-level manipulation localisation head, and modality gating. | Hybrid multi-modal, multi-task |
| `saff`         | Synchronisation-Aware Feature Fusion. Audio and visual token sequences attend to each other through cross-modal attention; a frame-aligned synchronisation score feeds the classifier. | Hybrid multi-modal (intermediate fusion) |
| `late_fusion`  | Independent audio and visual temporal encoders, combined by a learned decision-level weight. | Single-modal-style baseline (late fusion) |

### Why `saff_plus` is stronger

It folds the strongest ideas from the systems the review ranked highest into one
design:

- **Deeper fusion**: modality-specific transformer encoders feed several stacked
  bidirectional cross-modal attention blocks, instead of a single shallow fusion.
- **Graph attention (CM-GAN)**: a graph-attention layer reasons over visual and
  audio summary nodes, the relational mechanism behind the top-performing SAFF
  system in the review.
- **Offset-tolerant synchronisation**: a soft audio-visual alignment over the
  full time grid catches partial desynchronisation, not just frame-aligned
  mismatches.
- **Self-supervised auxiliary task**: cross-modal temporal feature prediction
  (predict one modality from the other), which the review links to better
  cross-dataset transfer.
- **Frame-level localisation**: a per-frame head identifies *which* segments are
  manipulated, not just whether the clip is fake.
- **Modality gating**: a learned gate down-weights an unreliable or missing
  modality for graceful degradation.

The model is multi-task: the training loss combines classification, a
synchronisation regulariser, the cross-modal prediction loss, and (when
frame-level labels are supplied) a localisation loss.

> This is a **runnable reference implementation**, not a model trained on real
> deepfake corpora. Weights are randomly initialised. `demo.py` runs the full
> pipeline on synthetic clips; `train.py` fits the model on labelled data.

## Inputs

The model consumes **per-frame features**, so heavy feature extraction stays
outside the model and the demo runs on CPU.

- `visual_features`: `FloatTensor (batch, num_frames, visual_dim)` — for example, per-frame CNN/ViT embeddings.
- `audio_features`: `FloatTensor (batch, num_frames, audio_dim)` — for example, per-frame MFCC or mel embeddings.
- `labels` (optional): `LongTensor (batch,)`, `0 = real`, `1 = fake`.

## Quickstart

```bash
pip install -r requirements.txt
python demo.py
```

### Load with `transformers`

Because the model uses custom code, load it with `trust_remote_code=True`:

```python
from transformers import AutoModel, AutoConfig
import torch

repo = "YOUR_USERNAME/hybrid-deepfake-detector"   # after you push
config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()

B, T = 1, config.num_frames
visual = torch.randn(B, T, config.visual_dim)
audio = torch.randn(B, T, config.audio_dim)

with torch.no_grad():
    out = model(visual_features=visual, audio_features=audio)

print(out.probs)         # P(real), P(fake)
print(out.sync_score)    # audio-visual synchronisation (saff and saff_plus)
print(out.frame_logits)  # per-frame manipulation logits (saff_plus only)
```

### Switch architecture

```python
from configuration_hybrid_deepfake import HybridDeepfakeConfig
from modeling_hybrid_deepfake import HybridDeepfakeForVideoClassification

cfg = HybridDeepfakeConfig(architecture="saff_plus")   # or "saff" or "late_fusion"
model = HybridDeepfakeForVideoClassification(cfg)
```

## Training on real data

`train.py` runs end-to-end on synthetic data so the loop is verifiable. To train
for real, replace `SyntheticDeepfakeDataset` with a loader that yields
per-frame visual and audio embeddings from a labelled corpus such as
FakeAVCeleb, DFDC, or AV-Deepfake1M:

```bash
python train.py --architecture saff --steps 200 --save_dir ./trained
```

## How the SAFF synchronisation signal works

Genuine clips show high frame-aligned correlation between the audio and visual
streams (lip motion matches phonemes). Forgeries with dubbed audio or imperfect
lip-sync break this correlation. The SAFF backbone computes the mean cosine
similarity between time-aligned audio and visual tokens after cross-modal
attention, and a light regulariser pushes the score high for real clips and low
for fakes. This is the cross-modal consistency cue the review identifies as the
most generalisable signal across generation methods.

## Intended use and limitations

- **Intended use**: research and education on multi-modal deepfake detection; a
  starting point for training on real corpora.
- **Not for production**: weights are random until you train them. Do not use
  for legal, forensic, or content-moderation decisions without rigorous
  evaluation on representative, demographically balanced data.
- **Known limitations** (from the review): cross-dataset generalisation gaps of
  20 to 45 percentage points, adversarial fragility, demographic bias in
  training data, and limited robustness to diffusion-generated content.

## Citation

```bibtex
@mastersthesis{fuseini2026deepfake,
  title  = {Single-Modal versus Hybrid Multi-Modal Frameworks for the Detection of Deepfake Videos: A Systematic Literature Review},
  author = {Fuseini, Mohammed Kamal-Deen},
  year   = {2026},
  note   = {MPhil in Cybersecurity and Digital Forensics}
}
```

## License

MIT.

"""Hybrid multi-modal deepfake detector (reference implementation).

Two architectures are provided behind a single Hugging Face model class:

  1. LateFusionBackbone  - the single-modal-style baseline. Independent audio
     and visual temporal encoders each produce a forgery score; the scores are
     combined at the decision level by a learned weighted sum.

  2. SAFFBackbone        - the synchronisation-aware cross-modal attention model
     that the review found to be the strongest paradigm. Visual and audio token
     sequences attend to each other, an explicit audio-visual synchronisation
     signal is computed, and the fused representation feeds a classifier head.

The public class HybridDeepfakeForVideoClassification selects the backbone from
the config and exposes a standard forward(visual_features, audio_features,
labels=None) interface returning logits, an optional loss, and the sync score.

Inputs
------
visual_features : FloatTensor (batch, num_frames, visual_dim)
audio_features  : FloatTensor (batch, num_frames, audio_dim)
labels          : LongTensor  (batch,) with 0 = real, 1 = fake   (optional)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from configuration_hybrid_deepfake import HybridDeepfakeConfig


@dataclass
class DeepfakeDetectorOutput(ModelOutput):
    """Standard output container so the model plays well with the HF Trainer."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    probs: Optional[torch.FloatTensor] = None
    sync_score: Optional[torch.FloatTensor] = None


def _masked_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean-pool over the temporal dimension (batch, T, d) -> (batch, d)."""
    return x.mean(dim=1)


# ---------------------------------------------------------------------------
# Baseline: late (decision-level) fusion
# ---------------------------------------------------------------------------
class LateFusionBackbone(nn.Module):
    def __init__(self, config: HybridDeepfakeConfig):
        super().__init__()
        h = config.hidden_dim

        self.visual_proj = nn.Linear(config.visual_dim, h)
        self.audio_proj = nn.Linear(config.audio_dim, h)

        self.visual_gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
        self.audio_gru = nn.GRU(h, h, batch_first=True, bidirectional=True)

        self.visual_head = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(2 * h, config.num_classes)
        )
        self.audio_head = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(2 * h, config.num_classes)
        )
        # learned decision-level weight between the two modalities
        self.fusion_logit = nn.Parameter(torch.zeros(1))

    def forward(self, visual_features, audio_features):
        v = F.relu(self.visual_proj(visual_features))
        a = F.relu(self.audio_proj(audio_features))

        v, _ = self.visual_gru(v)
        a, _ = self.audio_gru(a)

        v_logits = self.visual_head(_masked_mean(v))
        a_logits = self.audio_head(_masked_mean(a))

        w = torch.sigmoid(self.fusion_logit)  # scalar in (0, 1)
        logits = w * v_logits + (1.0 - w) * a_logits
        return logits, None


# ---------------------------------------------------------------------------
# SAFF: synchronisation-aware cross-modal attention fusion
# ---------------------------------------------------------------------------
class CrossModalBlock(nn.Module):
    """One block where each modality attends to the other (intermediate fusion)."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.v2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.a2v = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.v_norm = nn.LayerNorm(hidden_dim)
        self.a_norm = nn.LayerNorm(hidden_dim)
        self.v_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.a_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.v_ff_norm = nn.LayerNorm(hidden_dim)
        self.a_ff_norm = nn.LayerNorm(hidden_dim)

    def forward(self, v, a):
        # visual queries attend over audio keys/values, and vice versa
        v_att, _ = self.a2v(query=v, key=a, value=a)
        a_att, _ = self.v2a(query=a, key=v, value=v)
        v = self.v_norm(v + v_att)
        a = self.a_norm(a + a_att)
        v = self.v_ff_norm(v + self.v_ff(v))
        a = self.a_ff_norm(a + self.a_ff(a))
        return v, a


class SAFFBackbone(nn.Module):
    def __init__(self, config: HybridDeepfakeConfig):
        super().__init__()
        h = config.hidden_dim
        self.config = config

        self.visual_proj = nn.Linear(config.visual_dim, h)
        self.audio_proj = nn.Linear(config.audio_dim, h)

        self.pos = nn.Parameter(torch.randn(1, config.num_frames, h) * 0.02)

        self.blocks = nn.ModuleList(
            [CrossModalBlock(h, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )

        # fused representation = [pooled visual, pooled audio, sync feature]
        self.classifier = nn.Sequential(
            nn.Linear(2 * h + 1, h), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(h, config.num_classes),
        )

    def _sync_score(self, v, a):
        """Per-clip audio-visual synchronisation score.

        Genuine clips show high frame-aligned cross-modal correlation; forgeries
        with misaligned lip-sync or dubbed audio score lower. We use the mean
        cosine similarity between time-aligned visual and audio tokens.
        """
        v_n = F.normalize(v, dim=-1)
        a_n = F.normalize(a, dim=-1)
        per_frame = (v_n * a_n).sum(dim=-1)        # (batch, T)
        return per_frame.mean(dim=1, keepdim=True)  # (batch, 1)

    def forward(self, visual_features, audio_features):
        T = visual_features.size(1)
        v = self.visual_proj(visual_features) + self.pos[:, :T]
        a = self.audio_proj(audio_features) + self.pos[:, :T]

        for block in self.blocks:
            v, a = block(v, a)

        sync = self._sync_score(v, a)             # (batch, 1)
        fused = torch.cat([_masked_mean(v), _masked_mean(a), sync], dim=-1)
        logits = self.classifier(fused)
        return logits, sync


# ---------------------------------------------------------------------------
# Public Hugging Face model
# ---------------------------------------------------------------------------
class HybridDeepfakeForVideoClassification(PreTrainedModel):
    config_class = HybridDeepfakeConfig

    def __init__(self, config: HybridDeepfakeConfig):
        super().__init__(config)
        if config.architecture == "saff":
            self.backbone = SAFFBackbone(config)
        else:
            self.backbone = LateFusionBackbone(config)
        self.post_init()

    def forward(self, visual_features, audio_features, labels=None):
        logits, sync = self.backbone(visual_features, audio_features)
        probs = F.softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            # A light synchronisation regulariser: encourage high sync for real
            # clips (label 0) and low sync for fakes (label 1) when SAFF is used.
            if sync is not None and self.config.sync_weight > 0:
                target = 1.0 - labels.float().unsqueeze(1)  # real->1, fake->0
                sync_loss = F.mse_loss(torch.sigmoid(sync), target)
                loss = loss + self.config.sync_weight * sync_loss

        return DeepfakeDetectorOutput(loss=loss, logits=logits, probs=probs, sync_score=sync)

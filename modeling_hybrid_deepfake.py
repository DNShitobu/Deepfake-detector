"""Hybrid multi-modal deepfake detector (reference implementation).

Three interchangeable architectures sit behind one Hugging Face model class,
selected by the config ``architecture`` field:

  1. ``late_fusion`` - single-modal-style baseline. Independent audio and visual
     temporal encoders each produce a forgery score, combined at the decision
     level by a learned weight.

  2. ``saff`` - synchronisation-aware cross-modal attention. Visual and audio
     token sequences attend to each other, a frame-aligned synchronisation
     score feeds the classifier.

  3. ``saff_plus`` - the stronger architecture. It combines the strongest ideas
     from the systems the review ranked highest into one design:
       * modality-specific transformer self-attention encoders with learnable
         CLS tokens,
       * stacked bidirectional cross-modal attention blocks (deeper fusion),
       * a Cross-Modal Graph Attention layer (CM-GAN style) over modality
         summary nodes,
       * a soft temporal synchronisation module that tolerates audio-visual
         offsets,
       * an auxiliary temporal feature prediction task (predict one modality
         from the other) as a self-supervised signal,
       * a frame-level manipulation localisation head, and
       * modality gating for graceful degradation.

Inputs
------
visual_features : FloatTensor (batch, num_frames, visual_dim)
audio_features  : FloatTensor (batch, num_frames, audio_dim)
labels          : LongTensor  (batch,)            0 = real, 1 = fake   (optional)
frame_labels    : FloatTensor (batch, num_frames) 0 = real, 1 = fake   (optional)
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
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    probs: Optional[torch.FloatTensor] = None
    sync_score: Optional[torch.FloatTensor] = None
    frame_logits: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None


def _mean(x):
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
        self.visual_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(2 * h, config.num_classes))
        self.audio_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(2 * h, config.num_classes))
        self.fusion_logit = nn.Parameter(torch.zeros(1))

    def forward(self, visual_features, audio_features):
        v = F.relu(self.visual_proj(visual_features))
        a = F.relu(self.audio_proj(audio_features))
        v, _ = self.visual_gru(v)
        a, _ = self.audio_gru(a)
        v_logits = self.visual_head(_mean(v))
        a_logits = self.audio_head(_mean(a))
        w = torch.sigmoid(self.fusion_logit)
        logits = w * v_logits + (1.0 - w) * a_logits
        return {"logits": logits, "sync_score": None, "frame_logits": None, "aux_loss": None}


# ---------------------------------------------------------------------------
# Shared building block: one bidirectional cross-modal attention block
# ---------------------------------------------------------------------------
class CrossModalBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.v2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.a2v = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.v_norm = nn.LayerNorm(hidden_dim)
        self.a_norm = nn.LayerNorm(hidden_dim)
        self.v_ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.a_ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.v_ff_norm = nn.LayerNorm(hidden_dim)
        self.a_ff_norm = nn.LayerNorm(hidden_dim)

    def forward(self, v, a):
        v_att, _ = self.a2v(query=v, key=a, value=a)
        a_att, _ = self.v2a(query=a, key=v, value=v)
        v = self.v_norm(v + v_att)
        a = self.a_norm(a + a_att)
        v = self.v_ff_norm(v + self.v_ff(v))
        a = self.a_ff_norm(a + self.a_ff(a))
        return v, a


# ---------------------------------------------------------------------------
# SAFF: synchronisation-aware cross-modal attention (frame-aligned)
# ---------------------------------------------------------------------------
class SAFFBackbone(nn.Module):
    def __init__(self, config: HybridDeepfakeConfig):
        super().__init__()
        h = config.hidden_dim
        self.visual_proj = nn.Linear(config.visual_dim, h)
        self.audio_proj = nn.Linear(config.audio_dim, h)
        self.pos = nn.Parameter(torch.randn(1, config.num_frames, h) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossModalBlock(h, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * h + 1, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, config.num_classes)
        )

    def forward(self, visual_features, audio_features):
        T = visual_features.size(1)
        v = self.visual_proj(visual_features) + self.pos[:, :T]
        a = self.audio_proj(audio_features) + self.pos[:, :T]
        for block in self.blocks:
            v, a = block(v, a)
        v_n = F.normalize(v, dim=-1)
        a_n = F.normalize(a, dim=-1)
        sync = (v_n * a_n).sum(dim=-1).mean(dim=1, keepdim=True)
        fused = torch.cat([_mean(v), _mean(a), sync], dim=-1)
        logits = self.classifier(fused)
        return {"logits": logits, "sync_score": sync, "frame_logits": None, "aux_loss": None}


# ---------------------------------------------------------------------------
# Cross-Modal Graph Attention layer (CM-GAN style)
# ---------------------------------------------------------------------------
class GraphAttentionLayer(nn.Module):
    """Multi-head graph attention over a small set of fully connected nodes."""

    def __init__(self, in_dim, out_dim_total, num_heads, dropout):
        super().__init__()
        assert out_dim_total % num_heads == 0, "out_dim_total must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = out_dim_total // num_heads
        self.W = nn.Linear(in_dim, out_dim_total, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.1)
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):  # h: (B, N, in_dim)
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.num_heads, self.head_dim)   # (B, N, H, D)
        e_src = (Wh * self.a_src).sum(-1)                          # (B, N, H)
        e_dst = (Wh * self.a_dst).sum(-1)                          # (B, N, H)
        e = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))    # (B, N_i, N_j, H)
        alpha = self.dropout(torch.softmax(e, dim=2))             # attention over j
        out = torch.einsum("bijh,bjhd->bihd", alpha, Wh)          # (B, N, H, D)
        return out.reshape(B, N, self.num_heads * self.head_dim)


# ---------------------------------------------------------------------------
# SAFF++: the stronger architecture
# ---------------------------------------------------------------------------
class SAFFPlusBackbone(nn.Module):
    def __init__(self, config: HybridDeepfakeConfig):
        super().__init__()
        h = config.hidden_dim
        self.config = config

        self.visual_proj = nn.Linear(config.visual_dim, h)
        self.audio_proj = nn.Linear(config.audio_dim, h)

        # learnable CLS tokens + positional embeddings (room for the CLS slot)
        self.v_cls = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.a_cls = nn.Parameter(torch.randn(1, 1, h) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, config.num_frames + 1, h) * 0.02)

        # modality-specific self-attention encoders
        enc_layer_v = nn.TransformerEncoderLayer(h, config.num_heads, dim_feedforward=2 * h,
                                                 dropout=config.dropout, batch_first=True)
        enc_layer_a = nn.TransformerEncoderLayer(h, config.num_heads, dim_feedforward=2 * h,
                                                 dropout=config.dropout, batch_first=True)
        self.visual_encoder = nn.TransformerEncoder(enc_layer_v, num_layers=config.num_layers)
        self.audio_encoder = nn.TransformerEncoder(enc_layer_a, num_layers=config.num_layers)

        # deeper bidirectional cross-modal fusion
        self.cross_blocks = nn.ModuleList(
            [CrossModalBlock(h, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )

        # cross-modal graph attention over four summary nodes
        self.gat = GraphAttentionLayer(h, h, config.num_heads, config.dropout)

        # auxiliary temporal feature predictors (cross-modal forecasting)
        self.v2a_pred = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
        self.a2v_pred = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))

        # modality gating for graceful degradation
        self.gate = nn.Sequential(nn.Linear(2 * h, 2), nn.Softmax(dim=-1))

        # heads: graph-pooled (h) + gated CLS (h) + sync (1)
        self.classifier = nn.Sequential(
            nn.Linear(2 * h + 1, h), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(h, config.num_classes)
        )
        self.localiser = nn.Linear(h, 1)  # per-frame manipulation logit

    def _soft_sync(self, v_tok, a_tok):
        """Offset-tolerant synchronisation via soft cross-modal alignment."""
        v_n = F.normalize(v_tok, dim=-1)
        a_n = F.normalize(a_tok, dim=-1)
        sim = torch.bmm(v_n, a_n.transpose(1, 2))      # (B, T, T) cosine similarities
        align = torch.softmax(sim, dim=-1)             # soft alignment over audio frames
        aligned = (align * sim).sum(dim=-1)            # (B, T) best-aligned similarity
        return aligned.mean(dim=1, keepdim=True)       # (B, 1)

    def forward(self, visual_features, audio_features):
        B, T, _ = visual_features.shape
        v = self.visual_proj(visual_features)
        a = self.audio_proj(audio_features)

        v = torch.cat([self.v_cls.expand(B, -1, -1), v], dim=1) + self.pos[:, : T + 1]
        a = torch.cat([self.a_cls.expand(B, -1, -1), a], dim=1) + self.pos[:, : T + 1]

        v = self.visual_encoder(v)
        a = self.audio_encoder(a)
        for block in self.cross_blocks:
            v, a = block(v, a)

        v_cls, v_tok = v[:, 0], v[:, 1:]
        a_cls, a_tok = a[:, 0], a[:, 1:]

        # offset-tolerant synchronisation
        sync = self._soft_sync(v_tok, a_tok)           # (B, 1)

        # graph attention over four modality summary nodes
        nodes = torch.stack([v_cls, a_cls, _mean(v_tok), _mean(a_tok)], dim=1)  # (B, 4, h)
        graph = self.gat(nodes).mean(dim=1)            # (B, h)

        # modality gating between the two CLS summaries
        gate_w = self.gate(torch.cat([v_cls, a_cls], dim=-1))  # (B, 2)
        gated = gate_w[:, :1] * v_cls + gate_w[:, 1:] * a_cls  # (B, h)

        fused = torch.cat([graph, gated, sync], dim=-1)
        logits = self.classifier(fused)
        frame_logits = self.localiser(v_tok).squeeze(-1)        # (B, T)

        # auxiliary cross-modal temporal prediction loss (self-supervised)
        pred_a = self.v2a_pred(v_tok)
        pred_v = self.a2v_pred(a_tok)
        aux_loss = F.mse_loss(pred_a, a_tok.detach()) + F.mse_loss(pred_v, v_tok.detach())

        return {"logits": logits, "sync_score": sync, "frame_logits": frame_logits, "aux_loss": aux_loss}


# ---------------------------------------------------------------------------
# Public Hugging Face model
# ---------------------------------------------------------------------------
class HybridDeepfakeForVideoClassification(PreTrainedModel):
    config_class = HybridDeepfakeConfig

    def __init__(self, config: HybridDeepfakeConfig):
        super().__init__(config)
        if config.architecture == "saff_plus":
            self.backbone = SAFFPlusBackbone(config)
        elif config.architecture == "saff":
            self.backbone = SAFFBackbone(config)
        else:
            self.backbone = LateFusionBackbone(config)
        self.post_init()

    def forward(self, visual_features, audio_features, labels=None, frame_labels=None):
        out = self.backbone(visual_features, audio_features)
        logits = out["logits"]
        sync = out["sync_score"]
        frame_logits = out["frame_logits"]
        aux_loss = out["aux_loss"]
        probs = F.softmax(logits, dim=-1)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

            # synchronisation regulariser: real clips -> high sync, fakes -> low
            if sync is not None and self.config.sync_weight > 0:
                target = 1.0 - labels.float().unsqueeze(1)
                loss = loss + self.config.sync_weight * F.mse_loss(torch.sigmoid(sync), target)

            # auxiliary cross-modal prediction loss
            if aux_loss is not None and self.config.pred_weight > 0:
                loss = loss + self.config.pred_weight * aux_loss

            # optional frame-level localisation loss
            if frame_logits is not None and frame_labels is not None and self.config.loc_weight > 0:
                loss = loss + self.config.loc_weight * F.binary_cross_entropy_with_logits(
                    frame_logits, frame_labels.float()
                )

        return DeepfakeDetectorOutput(
            loss=loss, logits=logits, probs=probs,
            sync_score=sync, frame_logits=frame_logits, aux_loss=aux_loss,
        )

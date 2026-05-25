"""Configuration for the Hybrid Multi-Modal Deepfake Detector.

This config drives two interchangeable architectures studied in the systematic
literature review "Single-Modal versus Hybrid Multi-Modal Frameworks for the
Detection of Deepfake Videos":

  * "saff"        - synchronisation-aware cross-modal attention fusion
  * "late_fusion" - independent audio and visual branches combined at the
                    decision level (baseline)

The model consumes per-frame visual features and per-frame audio features
(for example, CNN/ViT frame embeddings and MFCC or mel embeddings), so the
heavy feature extraction stays outside the model and the demo runs on CPU.
"""

from transformers import PretrainedConfig


class HybridDeepfakeConfig(PretrainedConfig):
    model_type = "hybrid_deepfake"

    def __init__(
        self,
        architecture: str = "saff",
        visual_dim: int = 512,
        audio_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        num_frames: int = 32,
        num_classes: int = 2,
        dropout: float = 0.1,
        sync_weight: float = 0.5,
        **kwargs,
    ):
        if architecture not in ("saff", "late_fusion"):
            raise ValueError(
                f"architecture must be 'saff' or 'late_fusion', got {architecture!r}"
            )
        self.architecture = architecture
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.dropout = dropout
        self.sync_weight = sync_weight
        super().__init__(**kwargs)

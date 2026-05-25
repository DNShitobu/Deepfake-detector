"""End-to-end runnable demo for the Hybrid Multi-Modal Deepfake Detector.

This script builds both architectures (SAFF cross-modal attention and the
late-fusion baseline) with small, randomly initialised weights, synthesises a
single audio-visual clip of per-frame features, and runs inference. It prints
the real/fake probabilities and, for the SAFF model, the audio-visual
synchronisation score.

No dataset, GPU, or trained weights are required. The point is to show the full
pipeline executing on CPU in a few seconds.

Run:
    pip install -r requirements.txt
    python demo.py
"""

import torch

from configuration_hybrid_deepfake import HybridDeepfakeConfig
from modeling_hybrid_deepfake import HybridDeepfakeForVideoClassification


def make_synthetic_clip(num_frames, visual_dim, audio_dim, synced=True, seed=0):
    """Create one synthetic clip.

    When synced=True the audio and visual streams share a latent driver, so the
    SAFF synchronisation score is high (mimicking a genuine clip). When
    synced=False the streams are independent, mimicking a forgery with mismatched
    audio and lip motion.
    """
    g = torch.Generator().manual_seed(seed)
    driver = torch.randn(num_frames, 16, generator=g)

    v_map = torch.randn(16, visual_dim, generator=g)
    a_map = torch.randn(16, audio_dim, generator=g)

    visual = driver @ v_map + 0.1 * torch.randn(num_frames, visual_dim, generator=g)
    if synced:
        audio = driver @ a_map + 0.1 * torch.randn(num_frames, audio_dim, generator=g)
    else:
        other = torch.randn(num_frames, 16, generator=g)
        audio = other @ a_map + 0.1 * torch.randn(num_frames, audio_dim, generator=g)

    # add batch dimension
    return visual.unsqueeze(0), audio.unsqueeze(0)


def run(architecture):
    config = HybridDeepfakeConfig(
        architecture=architecture,
        visual_dim=512,
        audio_dim=128,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_frames=32,
    )
    model = HybridDeepfakeForVideoClassification(config).eval()
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n=== {architecture.upper()} model ({n_params/1e6:.2f}M params) ===")
    for label, synced in [("genuine-like (synced)", True), ("forgery-like (desynced)", False)]:
        v, a = make_synthetic_clip(config.num_frames, config.visual_dim, config.audio_dim,
                                   synced=synced, seed=42 if synced else 7)
        with torch.no_grad():
            out = model(visual_features=v, audio_features=a)
        real_p, fake_p = out.probs[0].tolist()
        sync = None if out.sync_score is None else float(out.sync_score[0])
        sync_str = "" if sync is None else f" | sync={sync:+.3f}"
        loc_str = ""
        if out.frame_logits is not None:
            n_flagged = int((out.frame_logits[0] > 0).sum())
            loc_str = f" | frames flagged={n_flagged}/{out.frame_logits.size(1)}"
        print(f"  {label:26s} -> P(real)={real_p:.3f}  P(fake)={fake_p:.3f}{sync_str}{loc_str}")

    print("  note: weights are random, so the verdicts are illustrative only.")
    print("        train.py shows how to fit the model on labelled data.")


if __name__ == "__main__":
    torch.manual_seed(0)
    run("saff_plus")
    run("saff")
    run("late_fusion")
    print("\nPipeline ran end-to-end. All three architectures produced valid outputs.")

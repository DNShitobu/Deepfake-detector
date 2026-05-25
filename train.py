"""Minimal, runnable training scaffold for the hybrid deepfake detector.

This script fits the model on a small batch of SYNTHETIC labelled clips so that
the full training loop runs end-to-end on CPU in seconds. The synthetic data is
constructed so that genuine clips have correlated audio-visual streams and
forgeries do not, which is the exact signal the SAFF model is designed to
exploit. Replace `SyntheticDeepfakeDataset` with a real loader (for example,
FakeAVCeleb or DFDC frame and audio embeddings) to train for real.

Run:
    pip install -r requirements.txt
    python train.py --architecture saff --steps 100
"""

import argparse

import torch
from torch.utils.data import Dataset, DataLoader

from configuration_hybrid_deepfake import HybridDeepfakeConfig
from modeling_hybrid_deepfake import HybridDeepfakeForVideoClassification


class SyntheticDeepfakeDataset(Dataset):
    """Genuine clips: shared audio-visual driver. Forgeries: independent streams."""

    def __init__(self, n=512, num_frames=32, visual_dim=512, audio_dim=128, seed=0):
        self.n = n
        self.num_frames = num_frames
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        g = torch.Generator().manual_seed(seed)
        self.v_map = torch.randn(16, visual_dim, generator=g)
        self.a_map = torch.randn(16, audio_dim, generator=g)
        self.labels = (torch.arange(n) % 2)  # alternating real(0)/fake(1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        g = torch.Generator().manual_seed(1000 + idx)
        driver = torch.randn(self.num_frames, 16, generator=g)
        visual = driver @ self.v_map + 0.1 * torch.randn(self.num_frames, self.visual_dim, generator=g)
        if label == 0:  # real -> synced
            audio = driver @ self.a_map + 0.1 * torch.randn(self.num_frames, self.audio_dim, generator=g)
        else:           # fake -> desynced
            other = torch.randn(self.num_frames, 16, generator=g)
            audio = other @ self.a_map + 0.1 * torch.randn(self.num_frames, self.audio_dim, generator=g)
        return {
            "visual_features": visual,
            "audio_features": audio,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def collate(batch):
    return {
        "visual_features": torch.stack([b["visual_features"] for b in batch]),
        "audio_features": torch.stack([b["audio_features"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", choices=["saff", "late_fusion"], default="saff")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save_dir", type=str, default=None,
                    help="If set, save the trained model here (Hugging Face format).")
    args = ap.parse_args()

    config = HybridDeepfakeConfig(architecture=args.architecture, hidden_dim=128, num_frames=32)
    model = HybridDeepfakeForVideoClassification(config).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ds = SyntheticDeepfakeDataset(n=args.batch_size * 32)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    step, running = 0, 0.0
    correct, total = 0, 0
    while step < args.steps:
        for batch in dl:
            out = model(**batch)
            out.loss.backward()
            opt.step()
            opt.zero_grad()

            running += float(out.loss)
            preds = out.logits.argmax(dim=-1)
            correct += int((preds == batch["labels"]).sum())
            total += len(preds)
            step += 1
            if step % 20 == 0:
                print(f"step {step:4d} | loss {running/20:.4f} | train_acc {correct/total:.3f}")
                running, correct, total = 0.0, 0, 0
            if step >= args.steps:
                break

    print("Training loop finished.")
    if args.save_dir:
        model.save_pretrained(args.save_dir)
        config.save_pretrained(args.save_dir)
        print(f"Saved model to {args.save_dir}")


if __name__ == "__main__":
    main()

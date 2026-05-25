# Publishing this model to your Hugging Face account

The model lives in this folder. Below are the exact steps to verify it locally
and push it to your own Hugging Face account. Your access token stays on your
machine and is never shared.

## 1. One-time setup

```bash
cd "deepfake-detector-model"
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]"
```

## 2. Verify it runs (optional but recommended)

```bash
python demo.py                 # runs both architectures on synthetic clips
python train.py --steps 60     # runs the training loop on synthetic data
```

You should see real/fake probabilities, a synchronisation score for the SAFF
model, and a falling training loss with rising accuracy.

## 3. Log in to Hugging Face

Create a write token at https://huggingface.co/settings/tokens (role: **Write**),
then:

```bash
huggingface-cli login          # paste the token when prompted; it is stored locally
```

## 4. Push to your account (the two key commands)

Replace `YOUR_USERNAME` with your Hugging Face username. The upload command
creates the repository automatically if it does not exist.

```bash
huggingface-cli upload YOUR_USERNAME/hybrid-deepfake-detector . --repo-type=model
```

That single command publishes the whole folder (model code, config, README
model card, demo and training scripts). To make it public, the repo defaults to
public; add `--private` if you want it private.

If you prefer the newer CLI syntax, the equivalent is:

```bash
hf auth login
hf upload YOUR_USERNAME/hybrid-deepfake-detector . --repo-type=model
```

## 5. Confirm

Open `https://huggingface.co/YOUR_USERNAME/hybrid-deepfake-detector`. The model
card renders from README.md. Others can then load it with:

```python
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "YOUR_USERNAME/hybrid-deepfake-detector", trust_remote_code=True
)
```

## Files in this repository

| File | Purpose |
|------|---------|
| `configuration_hybrid_deepfake.py` | Config class; selects `saff` or `late_fusion` |
| `modeling_hybrid_deepfake.py`      | Both architectures + HF model class |
| `config.json`                      | Default config with `auto_map` for `trust_remote_code` |
| `demo.py`                          | End-to-end runnable demo on synthetic clips |
| `train.py`                         | Runnable training loop (synthetic data; swap in a real loader) |
| `requirements.txt`                 | Python dependencies |
| `README.md`                        | Hugging Face model card |
| `.gitattributes`                   | Git LFS rules for future weight files |

## Notes

- Weights are randomly initialised. Train on a real corpus (FakeAVCeleb, DFDC,
  AV-Deepfake1M) before reporting any detection performance.
- After training with `python train.py --save_dir ./trained`, copy the produced
  `model.safetensors` into this folder and re-run the upload command to publish
  trained weights.

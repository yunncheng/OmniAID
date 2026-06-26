# Clean Test Scripts

Self-contained inference scripts for using OmniAID as a detector / reward model.
Drop `clean_test.py` (plus the two model files it imports) into any project
that already has `torch`, `torchvision`, `transformers`, and `PIL` installed and
it just works.

## 🛠️ Usage

### 1. Download the checkpoint
Download the pre-trained OmniAID v2 weights from Hugging Face:

| Backbone | HF filename | Native resolution |
|---|---|---|
| CLIP (v2) | [`ckpt/checkpoint_omniaid_v2.pth`](https://huggingface.co/Yunncheng/OmniAID/blob/main/ckpt/checkpoint_omniaid_v2.pth) | 336×336 |
| DINOv3 (v2) | [`ckpt/checkpoint_omniaid_dino_v2.pth`](https://huggingface.co/Yunncheng/OmniAID/blob/main/ckpt/checkpoint_omniaid_dino_v2.pth) | 448×448 |

The corresponding ckpt filenames are hardcoded in `clean_test.py` inside the
`BACKBONES` dict. If you want a different path, edit that dict.

### 2. Prepare your images
Modify the `image_paths` list in `main()` to point at local image files (PNG/JPG):

```python
image_paths = [
    'path/to/image1.png',
    'path/to/image2.jpg',
    # ...
]
```

### 3. Run the test
```bash
# CLIP / Mirage variant
python clean_test.py --backbone clip

# DINOv3 variant
python clean_test.py --backbone dino
```

The script prints a list of detection scores — **higher means more likely
AI-generated**.

## Files

| File | What it contains |
|---|---|
| `omniaid.py` | Self-contained `OmniAID` model class (CLIP backbone, hybrid MoE). Mirrors `models/OmniAID.py` with training-only logic (training modes, gradient checkpointing, orthogonality / balance losses, `requires_grad` bookkeeping) removed. Numerical output is bit-identical to the full implementation. |
| `omniaid-dino.py` | Self-contained `OmniAID_DINO` model class (DINOv3 backbone) plus the same training-logic simplifications as `omniaid.py`. Numerical output is bit-identical to `models/OmniAID_DINO.py`. |
| `clean_test.py` | Unified test harness for both backbones. Defines the `BACKBONES` registry (per-backbone ckpt path / native resolution / normalization stats), a single `process_images(...)` helper, and two scorer classes (`OmniAIDScorer` for CLIP, `OmniAIDDINOScorer` for DINOv3) sharing the same interface. CLI: `--backbone clip\|dino`. |

## Backbone registry

The single source of truth for backbone-specific settings is the `BACKBONES`
dict at the top of `clean_test.py`:

```python
BACKBONES = {
    "clip": {
        "ckpt_path":       "checkpoint_omniaid_v2.pth",
        "model_module":    "omniaid",
        "model_class":     "OmniAID",
        "image_resolution": 336,
        "mean":            [0.48145466, 0.4578275, 0.40821073],
        "std":             [0.26862954, 0.26130258, 0.27577711],
        "config_kwargs":   dict(
            CLIP_path="openai/clip-vit-large-patch14-336",
            num_experts=6, rank_per_expert=1, moe_top_k=2,
            moe_router_hidden_dim=256, is_hybrid=True,
        ),
    },
    "dino": {
        "ckpt_path":       "checkpoint_omniaid_dino_v2.pth",
        "model_module":    "omniaid-dino",
        "model_class":     "OmniAID_DINO",
        "image_resolution": 448,
        "mean":            [0.485, 0.456, 0.406],
        "std":             [0.229, 0.224, 0.225],
        "config_kwargs":   dict(
            DINOV3_path="facebook/dinov3-vitl16-pretrain-lvd1689m",
            num_experts=6, rank_per_expert=1, moe_top_k=2,
            moe_router_hidden_dim=256, is_hybrid=True,
        ),
    },
}
```

`process_images(image_paths, image_resolution, mean, std)` and both scorer classes
read from this dict, so adding a new backbone (e.g. a future CLIP-based v3) is
just a new dict entry.

## Integrating into a custom pipeline

Both scorers take a list of `PIL.Image` objects and return a list of floats:

```python
from clean_test import build_scorer

scorer = build_scorer("clip", device="cuda")  # or "dino"
scores = scorer(images)  # list[float], one per image, higher = more AI-like
```

`build_scorer(backbone)` is the recommended factory. If you want to instantiate
directly, both `OmniAIDScorer` and `OmniAIDDINOScorer` have the same constructor
signature `OmniAIDXxxScorer(device="cuda")`.
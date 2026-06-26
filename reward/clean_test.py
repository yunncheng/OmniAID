"""Self-contained test harness for OmniAID (CLIP) and OmniAID-DINO (DINOv3).

Each scorer is built from the matching simplified model in `omniaid.py` /
`omniaid-dino.py`. Use `--backbone clip|dino` to pick which one to test.

    python clean_test.py --backbone clip
    python clean_test.py --backbone dino
"""
import argparse
import importlib
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import types

from omniaid import OmniAID


# ---------- per-backbone config + preprocessing ----------

BACKBONES = {
    "clip": {
        "ckpt_path": "checkpoint_omniaid_v2.pth",
        "model_module": "omniaid",
        "model_class": "OmniAID",
        "image_resolution": 336,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std":  [0.26862954, 0.26130258, 0.27577711],
        "config_kwargs": dict(
            CLIP_path="openai/clip-vit-large-patch14-336",
            num_experts=6, rank_per_expert=1, moe_top_k=2,
            moe_router_hidden_dim=256, is_hybrid=True,
        ),
    },
    "dino": {
        "ckpt_path": "checkpoint_omniaid_dino_v2.pth",
        "model_module": "omniaid-dino",
        "model_class": "OmniAID_DINO",
        "image_resolution": 448,
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
        "config_kwargs": dict(
            DINOV3_path="facebook/dinov3-vitl16-pretrain-lvd1689m",
            num_experts=6, rank_per_expert=1, moe_top_k=2,
            moe_router_hidden_dim=256, is_hybrid=True,
        ),
    },
}


def process_images(image_paths, image_resolution, mean, std):
    """Single-resize preprocessing (v2 augmentation). Resolution/mean/std
    are explicit so the same helper handles CLIP and DINOv3."""
    tf = transforms.Compose([
        transforms.Resize((image_resolution, image_resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tensors = [tf(img) for img in image_paths]
    return torch.stack(tensors, dim=0)


# ---------- scorer factory ----------

def build_scorer(backbone: str, device: str = "cuda") -> "OmniAIDScorer":
    """Instantiate the right scorer for the given backbone."""
    if backbone == "clip":
        return OmniAIDScorer(device=device)
    if backbone == "dino":
        return OmniAIDDINOScorer(device=device)
    raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(BACKBONES)}.")


# ---------- scorers ----------

class OmniAIDScorer(nn.Module):
    """Scorer for the CLIP / Mirage backbone. Imports OmniAID from omniaid.py."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        spec = BACKBONES["clip"]
        cfg = types.SimpleNamespace(**spec["config_kwargs"])
        self.model = OmniAID(cfg)
        self._load_weights(spec["ckpt_path"])
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self._image_resolution = spec["image_resolution"]
        self._mean = spec["mean"]
        self._std = spec["std"]

    def _load_weights(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        cleaned = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in checkpoint.items()
        }
        self.model.load_state_dict(cleaned, strict=False)

    @torch.no_grad()
    def __call__(self, images):
        rewards = []
        for image in tqdm(images):
            x = process_images([image], self._image_resolution, self._mean, self._std).to(self.device)
            reward = self.model(x).detach().cpu().tolist()
            rewards.append(reward[0])
        return rewards


class OmniAIDDINOScorer(nn.Module):
    """Scorer for the DINOv3 backbone. Imports OmniAID_DINO from omniaid-dino.py."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        spec = BACKBONES["dino"]
        mod = importlib.import_module(spec["model_module"])
        model_cls = getattr(mod, spec["model_class"])
        cfg = types.SimpleNamespace(**spec["config_kwargs"])
        self.model = model_cls(cfg)
        self._load_weights(spec["ckpt_path"])
        self.model.to(device).eval()
        self.model.requires_grad_(False)
        self._image_resolution = spec["image_resolution"]
        self._mean = spec["mean"]
        self._std = spec["std"]

    def _load_weights(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        cleaned = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in checkpoint.items()
        }
        self.model.load_state_dict(cleaned, strict=False)

    @torch.no_grad()
    def __call__(self, images):
        rewards = []
        for image in tqdm(images):
            x = process_images([image], self._image_resolution, self._mean, self._std).to(self.device)
            reward = self.model(x).detach().cpu().tolist()
            rewards.append(reward[0])
        return rewards


# ---------- entry point ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", default="clip", choices=list(BACKBONES.keys()),
        help="Which OmniAID variant to test (clip=CLIP/Mirage, dino=DINOv3).",
    )
    args = parser.parse_args()

    scorer = build_scorer(args.backbone, device="cuda")

    image_paths = [
        'xx',
        'xx',
        # ... add your image paths here
    ]

    image_paths = [Image.open(p).convert('RGB') for p in image_paths]
    print(scorer(image_paths))


if __name__ == "__main__":
    main()
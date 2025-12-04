# OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2511.08423-B31B1B.svg)](https://arxiv.org/abs/2511.08423)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Yunncheng/OmniAID/tree/main)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Yunncheng/OmniAID-Demo)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/Yunncheng/Mirage-Test)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>


## 📖 Introduction

**OmniAID** is a universal AI-generated image detector designed for real-world, in-the-wild scenarios. 

Most existing detectors collapse under distribution shift because they entangle high-level semantic flaws (e.g., distorted humans, inconsistent object logic) and low-level generator artifacts (e.g., diffusion-specific fingerprints), learning a single fused representation that generalizes poorly.

To address these fundamental limitations, OmniAID explicitly decouples semantic and artifact cues through a **hybrid Mixture-of-Experts (MoE)** architecture—paired with a new modern dataset, Mirage, which reflects contemporary generative models and realistic threats.


![Method Framework](doc/method1.png)


## 🌟 Key Features

### 🧠 Hybrid MoE Architecture
- **Routable Semantic Experts**  
  Specialized experts dedicated to specific semantic domains (Human, Animal, Object, Scene, Anime).

- **Fixed Universal Artifact Expert**  
  Always active, focusing solely on **content-agnostic generative artifacts**.


### ⚙️ Two-Stage Training Strategy

1. **Expert Specialization**  
   Each semantic expert is trained independently with domain-specific hard sampling.

2. **Router Training**  
   A lightweight router learns to dispatch inputs to the most relevant semantic experts, while the artifact expert is always included.



## 🚀 Online Demo

Experience OmniAID instantly in your browser.
This demo is powered by the OmniAID checkpoint trained on Mirage-Train.

> **[Try OmniAID on Hugging Face Spaces](https://huggingface.co/spaces/Yunncheng/OmniAID-Demo)**

**Supported Modes:**
* **🤖 Auto (Router) Mode** (Default)
    The lightweight router dynamically analyzes the input image and assigns optimal weights to specific semantic experts.
* **🎛️ Manual Mode** (Analysis)
    Allows you to manually adjust expert weights to interpret how different semantic domains or the universal artifact expert contribute to the final detection score.


## 📚 Dataset


### 🔸 GenImage-SD v1.4 (Classified)
A reorganized subset of GenImage-SD v1.4, classified into semantic categories (Human_Animal, Object_Scene) to train the Semantic Experts.

[Download via Google Drive](https://drive.google.com/drive/folders/1Y5Fbf2Dm-trRxmmyXlcPgjYY9h_7BOUz?usp=sharing)

### 🔸 GenImage-SD v1.4 Reconstruction
The real images from the GenImage-SD v1.4 subset, reconstructed using the SD1.4 VAE.
We apply the reconstruction methodology from [AlignedForensics](https://github.com/AniSundar18/AlignedForensics/tree/master) to this specific dataset to serve as "purified" reference data for artifact learning.

[Download via Google Drive](https://drive.google.com/drive/folders/1c3Ybk4NEfAXDs4VyxRoT_MjHVz3nErrF?usp=sharing)

### 🔸 Mirage-Test
A challenging evaluation set containing images from held-out modern generators, optimized for high realism to rigorously test model generalization.

[Download via Hugging Face](https://huggingface.co/datasets/Yunncheng/Mirage-Test)  
[Download via Google Drive](https://drive.google.com/file/d/1-iGPbOkzGK-91LDyqeFSQefFPmZ41E2_/view?usp=sharing)


## 📦 Model Zoo

We provide pre-trained checkpoints trained on different datasets. All models use **CLIP-ViT-L/14@336px** as the backbone and are hosted on Hugging Face.

| Model Variant | Training Data | Filename | Download |
| :--- | :--- | :--- | :--- |
| **OmniAID (Recommended)** | **Mirage-Train** (Ours) | `checkpoint_mirage.pth` | [Link](https://huggingface.co/Yunncheng/OmniAID/blob/main/checkpoint_mirage.pth) |
| **OmniAID-GenImage** | GenImage-SD v1.4 | `checkpoint_genimage_sd14.pth` | [Link](https://huggingface.co/Yunncheng/OmniAID/blob/main/checkpoint_genimage_sd14.pth) |

> **Note:** 
> * **OmniAID (Recommended)** is trained on our Mirage-Train, offering the best generalization for real-world "in-the-wild" detection.
> * **OmniAID-GenImage** is trained on the standard academic dataset GenImage-SD v1.4, primarily for fair comparison with previous baselines.



## 🛠️ Installation

```bash
git clone https://github.com/yunncheng/OmniAID.git
cd OmniAID
pip install -r requirements.txt
```

## ⚡ Quick Start

To reproduce our results or train on your own data, please follow the steps below.

### 1. Configuration
Modify the configuration file `config.json` to set model hyperparameters (e.g., number of experts, rank, hidden dimensions) and other global settings.

```jsonc
{
    "CLIP_path": "openai/clip-vit-large-patch14-336",
    "num_experts": 3,
    "rank_per_expert": 4,
    // ...
}
```


### 2. Training
We provide a shell script for training. Before running, please open `scripts/train.sh` and configure the necessary paths:
* `DATA_PATH`: Path to your training dataset.
* `OUTPUT_DIR`: Directory where checkpoints will be saved.
* `LOG_DIR`: Directory where logs will be saved.
* `MOE_CONFIG_PATH`: Path to your `config.json`.

Once configured, start training:

```bash
bash scripts/train.sh
```

### 3. Evaluation
To evaluate the model on test sets, open `scripts/eval.sh` and set the following:
* `EVAL_DATA_PATH`: Path to the validation/test dataset.
* `OUTPUT_DIR`: Directory where results will be saved.
* `RESUME`: Path to the trained model weight (`.pth`).
* `MOE_CONFIG_PATH`: Path to your `config.json`.

Then run the evaluation script:

```bash
bash scripts/eval.sh
```

## 🙏 Acknowledgements
We gratefully acknowledge the outstanding open-source contributions that enabled this work.

### 🔸 Base Framework
Our main training/inference framework is developed on top of [AIDE](https://github.com/shilinyan99/AIDE/blob/main) and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt-V2). We sincerely thank the authors for their robust codebase.

### 🔸 Reconstruction Code
The reconstruction scripts located in `recon/` are adapted from [AlignedForensics](https://github.com/AniSundar18/AlignedForensics/tree/master). We are grateful to the authors for their valuable contribution to artifact purification research.



## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{guo2025omniaid,
  title={OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild},
  author={Guo, Yuncheng and Ye, Junyan and Zhang, Chenjue and Kang, Hengrui and Fu, Haohuan and He, Conghui and Li, Weijia},
  journal={arXiv preprint arXiv:2511.08423},
  year={2025}
}
```
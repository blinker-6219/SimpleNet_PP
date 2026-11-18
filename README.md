# 🤖🔧 SimpleNet_PP ➕➕

**Efficient transformer-based feature extraction backbones for industrial anomaly detection.**

Anomalies in industrial settings pose challenges in achieving accurate models while maintaining fast processing times. We propose improvements to SimpleNet using lightweight vision transformers as the backbone. This includes **RepViT** and **MobileViT**. The resulting RepViT-SimpleNet model was evaluated on the MVTec AD data-set.

--- 

## 📌 Table of Contents
1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Architecture](#-architecture)
4. [Project Structure](#-project-structure)
5. [Installation](#-installation)
6. [Dataset Setup](#-dataset-setup)
7. [Training](#-training)
8. [Evaluation & Benchmarking](#-evaluation--benchmarking)
9. [Results](#-results)
10. [Training Curves](#-training-curves)
11. [Qualitative Results](#-qualitative-results)
12. [Configuration Reference](#-configuration-reference)
13. [Known Limitations](#-known-limitations)
14. [Citation & Acknowledgements](#-citation--acknowledgements)

---

## 🔍 Overview
Memory-bank methods (SPADE, PaDiM, PatchCore) detect defects well but their memory and nearest-neighbour cost grows with the number of normal patches. SimpleNet removes the memory bank by training a small discriminator on normal vs. synthetic-anomaly features, but its quality is tied to a CNN backbone with mostly local receptive fields.

**SimpleNet_PP** keeps SimpleNet's lightweight feature-space detector and swaps the backbone for efficient transformer-style extractors that capture more global context at low cost.

## ✨ Key Features
- 🔌 **Drop-in backbones:** `repvit`, `mobilevit`, plus `resnet18` and `mobilenetv3` for baselines/ablations.
- 🧊 **Frozen, ImageNet-pretrained backbone**; only the feature adapter and discriminator are trained.
- 🎲 **Multi-seed protocol:** 5 seeds, mean ± std, paired t-test across the 15 MVTec categories.
- 📏 **Metrics:** image AUROC, pixel AUROC, PRO, parameter count, latency, FPS.
- 📈 **Progress logging** with tqdm and per-category result printing.

## 🏗️ Architecture
Pipeline: `Input → Frozen backbone (RepViT / MobileViT) → multi-stage feature aggregation → Feature adapter → Discriminator → Anomaly map + image score`.
During training only, Gaussian noise (σ = 0.015) is added to adapted features to create synthetic anomalies.

> 📷 **[PLACEHOLDER: architecture diagram]**
> Save as `assets/architecture/pipeline.png`, then use:
> `![Pipeline](assets/architecture/pipeline.png)`

> 📷 **[PLACEHOLDER: RepViT block / MobileViT block figures]**
> `assets/architecture/repvit_block.png`, `assets/architecture/mobilevit_block.png`

## 📁 Project Structure
```
rvsimplenet/
├── configs/default.yaml      # hyperparameters
├── models/
│   ├── backbone.py           # timm backbones (frozen)
│   └── simplenet.py          # adapter, discriminator, loss, scoring
├── utils/
│   ├── data.py               # MVTec AD loader
│   └── metrics.py            # I-AUROC, P-AUROC, PRO
├── scripts/
│   ├── aggregate.py          # mean±std + paired t-test
│   └── benchmark.py          # params / latency / FPS
├── assets/                   # figures for this README
│   ├── architecture/
│   ├── curves/
│   └── qualitative/
├── data/mvtec/               # YOU add the dataset here
├── runs/                     # JSON outputs (auto-created)
├── train.py
└── requirements.txt
```

## ⚙️ Installation
```bash
git clone <your-repo-url> && cd rvsimplenet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
A CUDA-capable GPU is strongly recommended.

## 🗂️ Dataset Setup
Download MVTec AD from the official site and extract so the 15 category folders sit directly under `data/mvtec/`:
```
data/mvtec/bottle/{train/good, test/<defect>, ground_truth/<defect>}
data/mvtec/cable/...
```

## 🚀 Training
```bash
# Quick sanity check (1 category, 1 seed)
python train.py --backbone repvit --cats bottle --seeds 0

# Full protocol
python train.py --backbone repvit
python train.py --backbone mobilevit
python train.py --backbone resnet18      # baseline, identical pipeline
```
Outputs: `runs/<backbone>/seed<N>.json`.

## 📊 Evaluation & Benchmarking
```bash
python scripts/aggregate.py runs/repvit runs/resnet18
python scripts/benchmark.py repvit 224
```

## 🔧 Configuration Reference
| Key | Default | Meaning |
|---|---|---|
| `backbone` | `repvit` | `repvit`, `mobilevit`, `resnet18`, `mobilenetv3` |
| `resize` / `imagesize` | 329 / 288 | resize, then center crop |
| `embed_dim` | 1536 | adapter / discriminator width |
| `sigma` | 0.015 | synthetic-anomaly noise std |
| `th` | 0.5 | truncated-L1 margin |
| `meta_epochs` / `gan_epochs` | 40 / 4 | training schedule |
| `batch_size` | 8 | |
| `lr_proj` / `lr_disc` | 1e-4 / 2e-4 | Adam learning rates |
| `wd` | 1e-5 | weight decay |
| `seeds` | 0–4 | repeated runs |

## ⚠️ Known Limitations
- Evaluated on MVTec AD only; no cross-dataset or real production-line validation.
- Reported metrics are final-epoch (no test-set checkpoint selection).
- FPS depends on GPU, resolution and batch size; always report these with results.
- Backbone stage indices in `models/backbone.py` are a design choice worth ablating.

## 📚 Citation & Acknowledgements
```bibtex
@misc{simplenetpp2026,
  title  = {SimpleNet++: Efficient Transformer based feature extraction backbones for Industrial Anomaly Detection},
  author = {Suruthi, C. J. and Srinivas, S. Karthick},
  year   = {2026},
  note   = {Amrita Vishwa Vidyapeetham, Coimbatore}
}
```
Built on SimpleNet (Liu et al., CVPR 2023), RepViT (Wang et al., 2023), MobileViT (Mehta & Rastegari, 2022), `timm`, and the MVTec AD dataset (Bergmann et al.).

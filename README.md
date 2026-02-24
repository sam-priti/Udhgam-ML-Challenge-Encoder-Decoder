# 🏆 Udhgam 2.0 ML Challenge — Team Encoder-Decoder

> **🥈 2nd Place Winner** — Udhgam 2.0 ML Challenge  
> CRNN-Attention architecture achieving **MAE: 0.14689** for prompt quality prediction from obfuscated tokens.

---

## 👥 Team: Encoder-Decoder

| Member | GitHub |
|--------|--------|
| **Sampriti Mohanty** | [@sam-priti](https://github.com/sam-priti) |
| **Himanshu Maurya** | [@HimanshuMaurya12](https://github.com/HimanshuMaurya12) |

---

## Overview

This repository contains the official solution for the **Udhgam 2.0 ML Challenge**. The task was to predict the quality score of LLM prompts based on **obfuscated token sequences** (integers) — without access to the original vocabulary or pre-trained embeddings.

**Challenge:** Predict prompt quality from scrambled/obfuscated integer tokens with no semantic context available.

---

## 🏅 Result

| Metric | Score |
|--------|-------|
| **Final MAE** | **0.14689** |
| **Placement** | **🥈 2nd Place** |

---

## Solution Architecture

Our approach uses a **CRNN-Attention Hybrid Model** trained entirely from scratch.

```
Obfuscated Token Sequence
        ↓
Learned Embeddings (dim=256)
        ↓
1D-CNN (local n-gram patterns)
        ↓
Bi-GRU (long-range dependencies)
        ↓
Attention Mechanism (critical token weighting)
        ↓
Dense Layer + Meta-Features (Length, Unique Ratio, Std Dev)
        ↓
Prompt Quality Score
```

### Key Components

**1. Learned Embeddings** — Fresh embedding layer (dim=256) trained from scratch, captures semantic relationships between obfuscated tokens without original vocabulary

**2. 1D-CNN** — Extracts local texture and n-gram patterns, detects specific phrasing structures in token space

**3. Bi-GRU** — Captures long-range dependencies and sequence flow in both directions

**4. Attention Mechanism** — Dynamically weights critical tokens, focuses on constraints or keywords

**5. Meta-Features** — Statistical features (Length, Unique Token Ratio, Std Dev) injected into final dense layer

---

## Optimization Techniques

- **Pseudo-Labeling** — High-confidence predictions used to expand training manifold
- **Test-Time Augmentation (TTA)** — Monte Carlo Dropout (10 inference passes) to reduce variance
- **Huber Loss** — `SmoothL1Loss` for stable convergence near zero error

---

## Project Structure

```
Udhgam-ML-Challenge-Encoder-Decoder/
├── Udhgam 2.0 MLC Encoder Decoder.ipynb   # Full solution notebook
├── Technical Report Encoder-Decoder.pdf    # Detailed technical report
├── requirements.txt                         # Dependencies
├── LICENSE                                  # MIT License
└── README.md                               # This file
```

---

## Setup

```bash
git clone https://github.com/sam-priti/Udhgam-ML-Challenge-Encoder-Decoder.git
cd Udhgam-ML-Challenge-Encoder-Decoder
pip install -r requirements.txt
```

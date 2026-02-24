# Udhgam 2.0 ML Challenge Solution 

**Team Name:** Encoder-Decoder  
**Score (MAE):** 0.14689

## Overview
This repository contains the source code for Team Encoder-Decoder's solution to the **Udhgam 2.0 ML Challenge**. The task was to predict the quality score of LLM prompts based on **obfuscated token sequences** (integers) without access to the original vocabulary or pre-trained embeddings.

## Solution Architecture
Our approach uses a **CRNN-Attention Hybrid Model** trained from scratch.

## Key Components:
1.  **Learned Embeddings:** We train a fresh embedding layer (`dim=256`) to capture semantic relationships between obfuscated tokens.
2.  **1D-CNN:** Extracts local texture and n-gram patterns (e.g., specific phrasing structures).
3.  **Bi-GRU:** Captures long-range dependencies and sequence flow.
4.  **Attention Mechanism:** Dynamically weights critical tokens, allowing the model to focus on constraints or keywords.
5.  **Meta-Features:** We inject statistical features (Length, Unique Ratio, Std Dev) into the final dense layer to provide a global context.

## Optimization Techniques:
* **Pseudo-Labeling:** Leveraged high-confidence predictions from our best model to expand the training manifold.
* **Test-Time Augmentation (TTA):** Implemented Monte Carlo Dropout (10 inference passes) to reduce variance and improve generalization.
* **Huber Loss:** Used `SmoothL1Loss` for stable convergence near zero error.

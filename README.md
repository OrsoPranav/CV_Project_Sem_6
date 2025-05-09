🧠 CNN-ViT Hybrid for Semi-Supervised Semantic Segmentation

This repository contains the official implementation of our semi-supervised semantic segmentation framework combining CNNs and Vision Transformers (ViTs) for robust, efficient, and scalable pixel-level predictions. It leverages both labeled and unlabeled data through pseudo-labeling, a novel loss function, and architectural enhancements like LoRA and FP16 optimization.

🚀 Project Overview

Semantic segmentation plays a critical role in scene understanding tasks across medical imaging, autonomous driving, and satellite vision. However, its performance heavily relies on large-scale labeled datasets, which are expensive and time-consuming to create. This project addresses that bottleneck through:

1) A hybrid encoder combining CNN's local feature capturing ability and ViT's global context modeling.
2) A semi-supervised training pipeline with pseudo-label generation and refinement.
3) A custom loss function to stabilize training with noisy pseudo labels.
4) Lightweight modifications such as Low-Rank Adaptation (LoRA) and mixed precision (FP16) to reduce compute demands.

📊 Results

The performance of our hybrid model under different configurations is presented below:

| Model                      | Parameters | mIoU (%) |
| -------------------------- | ---------- | -------- |
| CNN–ViT Hybrid Model       | 45.3M      | 73.8     |
| Hybrid Model + LoRA        | 38.7M      | 73.2     |
| Hybrid Model + FP16        | 45.1M      | 71.5     |
| Hybrid Model + LoRA + FP16 | 38.7M      | 69.9     |


These results demonstrate that LoRA significantly reduces parameters with only marginal accuracy loss, while FP16 offers speedups at the cost of performance.

Website:
![alt text](website.png)

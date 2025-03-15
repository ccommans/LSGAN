# LSGAN Project

This repository contains a reimplementation of the Least Squares Generative Adversarial Network (LSGAN) framework, along with extensive experiments on EMNIST, KTH-TIPS, and Tiny ImageNet datasets. Our implementation investigates different loss parameterizations and objective functions, evaluates the models using quantitative metrics (e.g., FID) and qualitative analyses (e.g., latent space interpolations), and includes a detailed analysis of training stability and image quality.

## Overview
Generative Adversarial Networks (GANs) are powerful generative models but suffer from issues such as vanishing gradients and mode collapse. LSGANs address these issues by replacing the conventional cross-entropy loss with a least squares loss. In this project, we re-implement LSGAN from scratch using PyTorch, experiment with different loss parameterizations, and evaluate the performance on multiple datasets:
- **EMNIST** (Handwritten letters)
- **KTH-TIPS** (Texture images)
- **Tiny ImageNet** (Animal images)

## Datasets and Models
### Datasets
Due to their large size, the datasets are not stored directly in this repository. Please download them from the following Google Drive links:
- **KTH-TIPS:** [Download KTH-TIPS Dataset](https://drive.google.com/drive/folders/17Ml5xawQWedCTiSzFeVRvZXVKKDFYfz2?usp=share_link)
- **Tiny ImageNet:** [Download Tiny ImageNet Dataset](https://drive.google.com/drive/folders/1CsfNecrZg4wboBHzkuZAOgQXX1zXX2Ed?usp=sharing)

Once downloaded, place them in the `data/` folder according to the following structure:
```
data/
├── emnist/
├── kth-tips/
└── tiny-imagenet-200/
```

### Final Models
The trained final models for each dataset are also hosted on Google Drive:
- **EMNIST Final Models:** [Download EMNIST Models](https://drive.google.com/drive/folders/1PCV2oo8pj99ANgA04VtkJqW3AhCft7oL?usp=share_link)
- **KTH-TIPS Final Models:** [Download KTH-TIPS Models](https://drive.google.com/drive/folders/1zBvRx_49-oM0UEoMmbFOxrfJ4A03RyxO?usp=share_link)
- **Tiny ImageNet Final Models:** [Download Tiny ImageNet Models](https://drive.google.com/drive/folders/1XeOBaHNZVRLokpmXztW--aQPNmW5vt1z?usp=share_link)

Download and place them in an appropriate folder (e.g., `model/emnist`, `model/kth`, `model/tiny`).

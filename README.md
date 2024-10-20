# HyperCMR
PyTorch implementation of the paper "[HyperCMR: Enhanced Multi-Contrast CMR Reconstruction with Eagle Loss](https://arxiv.org/abs/2410.03624)". This repository includes the code for our optimized Eagle-Loss function, designed to repair the under-sampled k-space high-frequency information of CMR Reconstruction.

[![Paper](https://img.shields.io/badge/Paper-Published-brightgreen.svg)](https://arxiv.org/abs/2410.03624)

[Presentation](https://www.youtube.com/watch?v=tBZq-MMEHPw)

# Test Datasets Results
Challenge: [MICCAI 2024 CMRxRecon Challenge](https://cmrxrecon.github.io/2024/Home.html)

We are Top 6 in Task 1: Multi-contrast CMR reconstruction

We are Top 5 in Task 2: Random sampling CMR reconstruction

![image](https://github.com/user-attachments/assets/b9ed68a7-d64b-41f2-8ab0-cd8e3319a34e)
<img width="1349" alt="image" src="https://github.com/user-attachments/assets/e9a78d5a-f920-42e2-b303-28b4af511ed3">
<img width="1355" alt="image" src="https://github.com/user-attachments/assets/5ef3a4ac-3f38-4f65-a895-4f78c43945bd">




## Abstract
Accelerating image acquisition for cardiac magnetic resonance imaging (CMRI) is a critical task. CMRxRecon2024 challenge aims to set the state of the art for multi-contrast CMR reconstruction. This paper presents HyperCMR, a novel framework designed to accelerate the reconstruction of multi-contrast cardiac magnetic resonance (CMR) images. HyperCMR enhances the existing PromptMR model by incorporating advanced loss functions, notably the innovative Eagle Loss, which is specifically designed to recover missing high-frequency information in undersampled k-space. Extensive experiments conducted on the CMRxRecon2024 challenge dataset demonstrate that HyperCMR consistently outperforms the baseline across multiple evaluation metrics, achieving superior SSIM and PSNR scores.
![image](https://github.com/user-attachments/assets/37538b80-5f3a-410c-851f-b07652198191)

## Data processing
```bash
python prepare_h5py_dataset_for_training.py
```

## Training
### Task1
```bash
python train_acc10.py
```
### Task2
on 4*A100 workstation
```bash
sbatch train_with_distribution.sbatch
```
## Acknowledgments
This repository was built on the following resources:
- Network: [PromptMR](https://github.com/hellopipu/PromptMR)
- Loss Baseline: [Eagle Loss](https://github.com/sypsyp97/Eagle_Loss)

## Citation
If you found this repository useful to you, please consider giving a star ⭐️ and citing our paper:
```bash
@article{xu2024hypercmr,
  title={HyperCMR: Enhanced Multi-Contrast CMR Reconstruction with Eagle Loss},
  author={Xu, Ruru and {\"O}zer, Caner and Oksuz, Ilkay},
  journal={arXiv preprint arXiv:2410.03624},
  year={2024}
}
```

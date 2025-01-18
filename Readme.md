# Tackling the Generative Learning Trilemma with Denoising Diffusion GANs

This project explores the paper **"Tackling the Generative Learning Trilemma with Denoising Diffusion GANs"** by Zhisheng Xiao, Karsten Kreis, and Arash Vahdat, presented at ICLR 2022 (Spotlight). The paper discusses the challenge of the generative learning trilemma, where generative models often struggle to balance high sample quality, mode coverage, and fast sampling. 

The authors propose a new model called **Denoising Diffusion GANs (DDGANs)**, which achieves impressive sample quality and diversity while reducing sampling time significantly, making it a promising model for real-world applications.


## Project Overview

We conducted multiple experiments to analyze and reproduce the results discussed in the paper. In our main notebook, `main.ipynb`, we evaluate the Denoising Diffusion GAN model and explore the implications of the proposed methods. We also further explore and visualize the denoising process and theory.

In addition to the original methods, we have included a script to run the project in environments with fewer than 8 GPUs, making it easier for those without access to distributed parallel setups.

We also provide a detailed report of our analysis and findings, which is included in this repository.

## Project Structure

- `main.ipynb`: Jupyter notebook for running experiments and analyzing results.
- `train_ddgan_1_gpu.py`: Implementation of the training script of Denoising Diffusion GAN model withoug torch distibuted
- `test_ddgan_with_plots`: Test script that plots intermediary generation steps.
- `report.pdf`: Detailed report of our analysis and experiments.
- `README.md`: This file with an overview of the project.

## Requirements

- NumPy
- Matplotlib
- tqdm
- scipy

Install the required dependencies using:

```bash
pip install -r requirements.txt

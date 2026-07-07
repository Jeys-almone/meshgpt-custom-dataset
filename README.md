<img src="./meshgpt.png" width="450px"></img>

# MeshGPT with a Custom Small-Scale Dataset

Implementation and experimental evaluation of **MeshGPT** using a custom dataset composed of real-world objects reconstructed through photogrammetry.

This repository extends the original implementation of MeshGPT by providing a complete experimental pipeline for dataset preparation, mesh processing, model training, mesh generation and quantitative evaluation.

---

## Original Project

This work is based on the original implementation developed by **Phil Wang (lucidrains)**:

https://github.com/lucidrains/meshgpt-pytorch

The original implementation provides the MeshGPT architecture proposed in:

> Siddiqui et al. *MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers*, 2023.

---

# Repository Objectives

The main objective of this repository is to reproduce and evaluate MeshGPT using a **small custom dataset** obtained from real objects.

The experimental workflow includes:

- image acquisition
- background removal
- photogrammetric reconstruction
- mesh augmentation
- mesh simplification
- MeshGPT training
- mesh generation
- quantitative evaluation
- comparison with Point-E

---

# Experimental Pipeline

```
Image Acquisition
        │
        ▼
Background Removal
        │
        ▼
Photogrammetry (Meshroom)
        │
        ▼
OBJ Mesh Export
        │
        ▼
Mesh Augmentation
        │
        ▼
Mesh Simplification
        │
        ▼
Mesh Autoencoder Training
        │
        ▼
Mesh Transformer Training
        │
        ▼
Mesh Generation
        │
        ▼
Evaluation
        │
        ▼
Comparison with Point-E
```

---

# Repository Structure

```
comparison/
dataset/
dataset_sem_fundo/
evaluation/
generated_meshes/
meshes/
meshes_augmented/
meshes_simplified/
meshgpt_pytorch/
models/
scripts/
tests/
```

A detailed description of each folder is presented in the next sections.

---

# Installation

Clone the repository

```bash
git clone https://github.com/Jeys-almone/meshgpt-custom-dataset.git

cd meshgpt-custom-dataset
```

Create the environment

```bash
conda create -n meshgpt python=3.10

conda activate meshgpt
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Experimental Workflow

The complete experimental pipeline adopted in this work is divided into the following stages.

1. Dataset preparation

2. Background removal

3. 3D reconstruction using Meshroom

4. Mesh augmentation

5. Mesh simplification

6. Autoencoder training

7. Transformer training

8. Mesh generation

9. Quantitative evaluation

10. Comparison with Point-E

Each step is described in detail throughout this documentation.

---

# Results

The repository contains:

- Original images
- Background-free images
- Reconstructed meshes
- Augmented meshes
- Simplified meshes
- Trained models
- Generated meshes
- Quantitative evaluation scripts

The evaluation employs:

- Chamfer Distance
- Minimum Matching Distance (MMD)
- Coverage (COV)
- 1-Nearest Neighbor Accuracy (1-NNA)

---

# Acknowledgments

This project is based on the original MeshGPT implementation developed by Phil Wang (lucidrains).

We also acknowledge the authors of MeshGPT for making their work publicly available.

---

# Citation

If you use this repository, please cite the original MeshGPT paper.

```bibtex
@inproceedings{Siddiqui2023MeshGPTGT,
    title   = {MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers},
    author  = {Yawar Siddiqui and Antonio Alliegro and Alexey Artemov and Tatiana Tommasi and Daniele Sirigatti and Vladislav Rosov and Angela Dai and Matthias Nie{\ss}ner},
    year    = {2023}
}
```

---

# Authors

Jeysraelly Almone

Universidade Federal do Maranhão (UFMA)

Research project developed for the evaluation of MeshGPT using a custom small-scale dataset.

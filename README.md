# Dual Deep Mesh Prior [ECCV2022]

### [Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630358.pdf) | [Supp](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136630358-supp.pdf) | [Poster](https://drive.google.com/file/d/1NS-2wkIeMXFGOlP778LojQLFSIHNMu8r/view?usp=sharing)

The official implementation of **Learning Self-prior for Mesh Denoising using Dual Graph Convolutional Networks**, [ECCV2022](https://eccv2022.ecva.net/program/accepted-papers/#:~:text=Yu%20(ETH%20Zurich)-,4934,-Learning%20Self%2Dprior).

A deep-learning framework for mesh denoising from a single noisy input, where two graph convolutional networks are trained jointly to filter vertex positions and facet normals apart.

<img src="fig/anim.gif" align="top" width="400">

## Method Overview

<img src="fig/overview.png">

## Results

<img src="fig/representitive.png">

___

## Getting Started

### 0. Environments

<img src="https://img.shields.io/badge/GPU-NVIDIA_GeForce_RTX_4070_12GB-blue" alt="NVIDIA GeForce TITAN X 12GB">

```
python==3.10
torch==1.13.1
torch-geometric==2.2.0
```

### 1. Installation
```
git clone https://github.com/astaka-pe/Dual-DMP
cd Dual-DMP
docker image build -t astaka-pe/ddmp .
docker run -itd --gpus all -p 8080:8080 --name ddmp -v .:/work astaka-pe/ddmp
docker exec -it ddmp /bin/bash
```
<!-- conda env create -f environment.yml
conda activate ddmp -->

### 2. Preparation

The Dataset is distributed as a zip file. Please unzip and place it under Dual-DMP directory. 

```
unzip datasets.zip
```

### 3. Training

- CAD model

```
python3 main.py -i datasets/fandisk --k1 3 --k2 0 --k3 3 --k4 4 --k5 2 --bnfloop 5
```

- Non-CAD model
```
python3 main.py -i datasets/ankylosaurus
```

- Real-scanned model
```
python3 main.py -i datasets/pyramid --iter 50
```

You can monitor the training progress through the web viewer. (Default: http://localhost:8080)

<img src="fig/viewer.gif" width="800">

Outputs will be generated under `datasets/{model-name}/output/` with their MAD scores.

___
## Appendix
### Training with your own data
Place a noisy mesh and a ground-truth mesh under `datasets/{model-name}/` .
- Noisy mesh: `{model-name}_noise.obj`
- Ground-truth mesh: `{model-name}_gt.obj`

Run 
```
python3 preprocess/preprocess.py -i datasets/{model-name}
```
for edge-based normalization and creating initial smoothed mesh.

Finally, run
```
python3 main.py -i datasets/{model-name}
```
You should set appropriate weights as discribed in the paper.

### Training without using ground-truth data
After runnning `preprocess.py`, run
```
python3 main4real.py -i datasets/{model-name}
```

### Creating noisy data
Run
```
python3 preprocess/noisemaker.py -i datasets/{model-name}/{model-name}.obj --level {noise-level}
```
___

## Citation
```
@InProceedings{hattori2022ddmp,
  author        = {Hattori, Shota and Yatagawa, Tatsuya and Ohtake, Yutaka and Suzuki, Hiromasa},
  title         = {Learning Self-prior for Mesh Denoising using Dual Graph Convolutional Networks},
  booktitle     = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year          = {2022},
  doi           = {10.1007/978-3-031-20062-5_21}
}
```

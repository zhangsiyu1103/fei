# Beyond Output Faithfulness: Learning Attributions that Preserve Computational Pathways

Official code repository for "Beyond Output Faithfulness: Learning Attributions that Preserve Computational Pathways"

## Overview

This repository implements methods for generating faithful attribution maps for neural networks while preserving the computational pathways used by the model. The approach uses an ensemble-based optimization framework with multiple activation preservation mechanisms to ensure robustness and faithfulness of attributions.

## Key Features

- **Multiple Defense Mechanisms**: IBM, AVM, IVM, VM modes for gradient-based defenses
- **Ensemble Attribution Generation**: Generates attributions using multiple area constraints (0.1, 0.3, 0.5, 0.7, 0.9)
- **Comprehensive Evaluation**: Insertion/deletion metrics, intermediate layer analysis, and faithfulness evaluation
- **Multi-Dataset Support**: ImageNet, CUB-200, CIFAR-10, COCO, and VOC
- **Multi-Architecture Support**: VGG (16/19), ResNet (18/34/50/101), GoogLeNet, and AlexNet

## Installation

### Requirements

- Python 3.7+
- PyTorch >= 1.8.0
- torchvision
- numpy
- scipy
- opencv-python
- matplotlib
- tqdm
- torchray
- cub_tools (for CUB-200 dataset)

### Environment Variables

Set the following environment variables before running the code:

```bash
export IMAGENETDIR=/path/to/imagenet
export CUBDIR=/path/to/CUB_200_2011
export PRETRAINED=/path/to/pretrained/models
```

```

## Usage

### Basic Attribution Generation

Generate attributions for a model with a specific defense mode:

```bash
python eval.py \
    --model vgg16 \
    --dataset imagenet \
    --defense_mode IBM \
    --save_dir ./results
```

### Available Options

- `--model`: Model architecture (vgg16, vgg19, resnet18, resnet34, resnet50, resnet101, googlenet, alexnet)
- `--dataset`: Dataset to use (imagenet, cub, cifar10)
- `--defense_mode`: Defense mechanism (IBM, VM, IVM, AVM, NONE)
- `--save_dir`: Directory to save results
- `--visualize`: Enable visualization of attributions


### Saving Attributions

To generate and save attribution maps for later evaluation:

```bash
python eval_save.py \
    --model resnet50 \
    --dataset imagenet \
    --defense_mode IBM \
    --save_dir ./result_save
```

### Faithfulness Evaluation

Evaluate saved attributions using perturbation metrics:

```bash
python faithful_eval.py \
    --model resnet50 \
    --dataset imagenet \
    --attr_dir ./result_save \
    --save_dir ./result_faithful \
    --batch_size 4
```

### Intermediate Layer Analysis

Analyze intermediate layer activations to evaluate computational pathway preservation:

```bash
python internal_eval.py \
    --model vgg16 \
    --dataset imagenet \
    --attr_dir ./result_save \
    --save_dir ./result_internal_eval \
    --metrics mse overlap correlation cosine \```

Available metrics:
- `mse`: Mean squared error between activations
- `overlap`: Unit activation overlap (activated/inactivated neurons)
- `correlation`: Activation correlation
- `cosine`: Cosine similarity between activations
```

### Evaluation Metrics

## Perturbation Metrics

The code implements insertion and deletion game metrics with multiple substrates:

- **Insertion**: Gradually insert high-attribution pixels
- **Deletion**: Gradually remove high-attribution pixels

Metrics compute Area Under Curve (AUC) scores to evaluate attribution quality.

## Intermediate Layer Metrics

- **MSE**: Mean squared error between original and perturbed activations
- **Unit Overlap**: Ratio of activated/inactivated neurons that remain in the same state
- **Correlation**: Pearson correlation between activation patterns
- **Cosine Similarity**: Cosine similarity between activation vectors

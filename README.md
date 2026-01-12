# Custom GCViT Image Classifier

## Index

- [Tech Stack](#tech-stack)
- [Overview](#overview)
- [Quick Start](#Quick-Start)
- [Weights and Initialization](#Weights-and-Initialization)
- [Training Details](#Training-Details)
- [Perspective](#Perspective)
- [Acknowledgements](#Acknowledgements)

## Tech stack

<a href="https://www.python.org/">
  <img src="https://www.moosoft.com/wp-content/uploads/2021/07/Python.png" alt="HTML" width="75" height="75">
</a>

<a href="https://www.tensorflow.org/">
  <img src="https://miro.medium.com/v2/resize:fit:256/1*cKG1LJvVTaWqSkYSyVqtsQ.png" alt="Tensorflow" width="75" height="75">
</a>

<a href="https://numpy.org/">
  <img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg" alt="Numpy" width="115" height="75">
</a>

## Overview

This repository contains an end-to-end image classification pipeline built around GCViT (Global Context Vision Transformer) models in TensorFlow/Keras. The project emphasizes empirical ML practice, including dataset handling, reproducible training configurations, and practical deployment considerations.

While originally developed for fine-grained mushroom species classification, the codebase is structured to support reproducible experimentation and model benchmarking across datasets. 

Project Goals

Train and fine-tune GCViT models for image classification

Support reproducible experiments via explicit configuration files

Handle real-world data challenges (noise, class imbalance, scale)

Enable deployment-oriented workflows (e.g., TFLite export)

Provide a toy, reviewer-friendly run that executes end-to-end with minimal setup

Repository structure:

gcvit/
├── layers/          # GCViT building blocks (attention, embedding, blocks, etc.)
├── model/           # GCViT model definition
├── training/        # Data loading, preprocessing, utilities
├── main.py          # Training entrypoint
configs/
├── config.py        # Full dataset / research configuration
├── config_toy_cifar10.py  # Lightweight toy configuration
assets/
├── checkpoints/     # Saved training checkpoints
├── models/          # Saved Keras / TFLite models
├── training_performance/ # Training curves and metrics


references:

- [The model architecture](./model/gcvit.py)
- [The blocks used in architecture](./model/blocks.py)
- [A training script](./model/main.py)
- [Some helper functions](./model/util.py)
- [Some functions specific to the data pipeline](./model/data.py)

## Quick Start

### installation

> git clone https://github.com/r-dug/GCViT_Classifier.git
> cd GCViT_Classifier
> python3 -m venv .venv
> pip install -e .

This installation method enables imports and exposes the training entrypoint

### Run toy config

running this command will allow a user to test the environment on the CIFAR10 dataset (with images resized to fit the GCViT dimensionality / window sizes)

> mv ./gcvit/training/toy_config.py ./gcvit/training/config.py
> gcvit-train

The full config expects a directory-based image dataset and mirrors the interface of tf.keras.preprocessing.image_dataset_from_directory.

## Weights and Initialization

Training proceeds as follows:

If a local fine-tuning checkpoint exists, it is loaded.

Otherwise, pretrained ImageNet weights are downloaded automatically from:

https://github.com/awsaf49/gcvit-tf/releases


Weights are loaded with skip_mismatch=True to allow adaptation to new class counts.

This ensures the code is runnable out-of-the-box while supporting iterative fine-tuning.

## Training Details

Loss: CategoricalCrossentropy

Metrics: accuracy

Optimization: Adam / AdamW (configurable)

Training strategy: staged unfreezing of transformer layers

Reproducibility: fixed random seeds in configs

## Perspective

This project reflects my approach to empirical ML research:

Prioritizing data quality and interface consistency

Designing pipelines that support rapid experimentation

Explicit handling of model constraints and failure modes

Balancing accuracy with deployability and usability

While not security-focused itself, the same principles apply to robustness and safety-oriented ML systems.

## Acknowledgements

A special thank you to [Dr. Hichem Figui](https://engineering.louisville.edu/faculty/hichem-frigui/), who generously offered his time and expertise in computer vision to provide guidance. After some discussion about my personal project to build a mobile application to identify ascocarps e encouraged me to explore newer model architectures (nudging me, specifically towards the use of transformers) and to focus on one part of the problem at a time, with accuracy taking priority over model size. I have to agree. What good is our model's size if it can't perform reasonably well?


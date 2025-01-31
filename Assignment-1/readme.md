# VAE and Autoencoder MNIST Implementation

PyTorch implementation of Variational Autoencoder (VAE) and standard Autoencoder for MNIST dataset, based on MIT's 6.S978 course assignment.

## Requirements

* Python 3.8+
* PyTorch 1.8+
* torchvision
* numpy
* matplotlib
* tqdm
* CUDA-capable GPU (optional)

## Quick Start

1. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib tqdm
```

2. Run implementations:
```bash
# For VAE
python vae.py

# For standard autoencoder
python autoencoder.py
```

## Features

* Two autoencoder architectures:
  * Variational Autoencoder with SGVB and KL-WO-E regularization
  * Convolutional Autoencoder with skip connections

* Visualization tools:
  * Original vs reconstructed image comparison
  * 2D latent space visualization
  * Training loss curves

## Key Parameters

Modify these in Config class:
```python
input_dim = 784  # MNIST image size
hidden_dims = [128, 32, 16, 4]  # Network architecture
batch_size = 256
epochs = 50
lr = 1e-3
```

## Results

* Both implementations include:
  * Training progress display with tqdm
  * Loss tracking per epoch
  * Image reconstruction visualization
  * Latent space exploration

## Attribution

Based on Assignment 1 from MIT's 6.S978 course: https://mit-6s978.github.io/schedule.html

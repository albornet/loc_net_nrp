# Haptics predictive coding and 3D localisation

The 3D localisation of a human arm is decoded from the latent activity evoked by an array of haptic signals in a predictive coding network

## Installation

Install the following packages in your environment
- torch
- torchvision
- numpy
- pandas
- matplotlib
- imageio

## Usage

To train your network
 - Set the model parameters and the dataset path in train_net.py
 - Run training in terminal
```bash
python train_net.py
```

To test a trained network
 - Define the model you want to load for testing in test_net.py
 - Run the test in terminal
```bash
python test_net.py
```

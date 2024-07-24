# V-BeachNet

This repository contains the official PyTorch implementation for the paper "A New Framework for Quantifying Alongshore Variability of Swash Motion Using Fully Convolutional Networks." V-BeachNet is built upon V-FloodNet.

**V-BeachNet paper:**  
Salatin, R., Chen, Q., Raubenheimer, B., Elgar, S., Gorrell, L., & Li, X. (2024). A New Framework for Quantifying Alongshore Variability of Swash Motion Using Fully Convolutional Networks. Coastal Engineering, 104542.

**V-FloodNet paper:**  
Liang, Y., Li, X., Tsai, B., Chen, Q., & Jafari, N. (2023). V-FloodNet: A video segmentation system for urban flood detection and quantification. Environmental Modelling & Software, 160, 105586.

## Prerequisites

Ensure you have Conda installed on your Linux system with Python 3.11.6 and Nvidia GPU driver installed. You can install it from [here](https://docs.anaconda.com/anaconda/install/linux/).

## Steps

1. Clone this repository and change directory:
```sh
git clone https://github.com/rezasalatin/V-BeachNet.git
cd V-BeachNet
```

2. Create the virtual environment
```sh
conda env create -f environment.yml
conda activate vbeach
pip install -r requirements.txt
```

3. Visit the "Training_Station" folder and copy your manually segmented dataset to this directory. Run the following command to train the model:

```sh
./train_video_seg.sh
```

4. Visit the "Testing_Station" folder and copy your data to this directory. Run the following command to automatically segment your data:

```sh
./test_video_seg.sh
```
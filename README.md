# V-BeachNet

This repository contains the official PyTorch implementation for the paper "A New Framework for Quantifying Alongshore Variability of Swash Motion Using Fully Convolutional Networks." V-BeachNet is built upon V-FloodNet.

**V-BeachNet paper:**  
Salatin, R., Chen, Q., Raubenheimer, B., Elgar, S., Gorrell, L., & Li, X. (2024). A New Framework for Quantifying Alongshore Variability of Swash Motion Using Fully Convolutional Networks. Coastal Engineering, 104542.

**V-FloodNet paper:**  
Liang, Y., Li, X., Tsai, B., Chen, Q., & Jafari, N. (2023). V-FloodNet: A video segmentation system for urban flood detection and quantification. Environmental Modelling & Software, 160, 105586.

## Prerequisites

This code is tested on a newly installed Ubuntu 24.04 with default version of Python and Nvidia GPU.

1. Install Anaconda prerequisite (Can also be accessed from [here](https://docs.anaconda.com/anaconda/install/linux/)):
```sh
sudo apt update && \
sudo apt install libgl1-mesa-dri libegl1 libglu1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2-data libasound2-plugins libxi6 libxtst6
```

2. Download Anaconda3:
```sh
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
```

3. Located the downloaded file and install it:
```sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh
```

## Steps

1. Clone this repository and change directory:
```sh
git clone https://github.com/rezasalatin/V-BeachNet.git
cd V-BeachNet
```

2. Create the virtual environment with the requirements:
```sh
conda env create -f environment.yml
conda activate vbeach
```

3. Visit the "Training_Station" folder and copy your manually segmented (using [labelme](https://github.com/labelmeai/labelme)) dataset to this directory. Open the following file to change any of the variables and save it. Then execute it to train the model:
```sh
./train_video_seg.sh
```
Access your trained model from log/ directory.

4. Visit the "Testing_Station" folder and copy your data to this directory. Open the following file to change any of the variables (especially model path from the log/ folder) and save it. Then execute it to test the model:
```sh
./test_video_seg.sh
```
Access your segmented data from output directory.

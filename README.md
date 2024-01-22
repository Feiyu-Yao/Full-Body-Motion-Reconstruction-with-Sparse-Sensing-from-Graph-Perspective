# Full-Body-Motion-Reconstruction-with-Sparse-Sensing-from-Graph-Perspective
The pytorch implementation for AAAI2024 paper "Full-Body Motion Reconstruction with Sparse Sensing from Graph Perspective".


Datasets
----------
1. Download the dataset AMASS from [AMASS](https://amass.is.tue.mpg.de/index.html).
2. Download the body model http://smpl.is.tue.mpg.de from and placed them in `support_data/body_models` directory of this repository.
3. Run `prepare_data.py` to prepare data from VR device. The data is split referring to the folder `data_split`.

License and Acknowledgement
----------
This project is released under the MIT license. We refer to the code framework in  [AvatarPoser](https://github.com/eth-siplab/AvatarPoser) and [SCI-NET](https://github.com/cszn/KAIR) for network training. 

# -- Multivariate Anomaly Detection for Time Series Data with GANs -- #

# MAD-GAN

This repository tries to replicate the paper, _[MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial Networks](https://arxiv.org/pdf/1901.04997.pdf)_, by Dan Li, Dacheng Chen, Jonathan Goh, and See-Kiong Ng, and adapt the proposed approach for anomaly detection on image sequences.
MAD-GAN architecture was reimplemented on PyTorch, paper experiments were reproduced. Some of the obtained metrics are similar, while some differs noticeably. It can be connected with differences in implementation of controversial moments (issues from base paper git repo) and used hyperparameters.

As a follow-up MAD-GAN implementation was modifies for working with image sequences, using addition CNN layers. Modified model showed 65% accuracy and 88% Precision on single number data (train dataset consist of one number, and test of this number + anomalies inside). Directions for future work are to tune the hyperparameters and model architecture for improving current results on image sequences and succeeding in processing all MNIST numbers data sequences.

## Quickstart

- Python3

- install packages listed in requirements.txt

- Download the datasets and copy them to ./data folder

- To train the models on different data sets and to replicate the experiments:
  
  """python main.py"""

- Experiments results are saved to ./experiments_results folder

- The samples of generator's generation results are saved to ./generation_results folder

- To train MNIST data encoder (is needed for MNIST sequences generation):
    
    """python mnist_encoder.py""""

- To generate MNIST sequences:

    """python MNIST_data_generation.py"""

## Data

We evaluate the method on the KDD99, SWaT and WADI datasets, however, they are not uploaded in this repository.
They could be downloaded from https://drive.google.com/drive/folders/1yDV4rccrHEpoukrbDO_QJXd2XJoWBZej?usp=sharing

You can also download (or request) the original data at http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html and https://itrust.sutd.edu.sg/

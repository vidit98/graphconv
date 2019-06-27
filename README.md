## Introduction

![Semantic Segmentation](https://ibb.co/8bmbmxM)


Semantic segmentation is the task of assigning each pixel a class. Many different methods are proposed for the same. This a pytorch implementation of [this](https://papers.nips.cc/paper/8135-beyond-grids-learning-graph-representations-for-visual-recognition.pdf) paper. To know in more detail about the implementation and method refer to my blog post(link to be added).

## Requirements
* pytorch 1.1
* visdom
* python3


## Highlights

* Model can be distributed over many GPUs: pytorch provides the facility to divide data into various GPUs and train the network parallely. But here you can divide the model in various GPUs.
* Live Data Visualization: You can visualize the training curve live using visdom.
* Flexible to remove and add GCU units.
## TO DO
- [ ] Give flexiblity to use number of GPUs or to run only on CPU, currently model uses 2 GPUs.
- [ ] Upload checkpoint files
- [ ] Optimize code

## Training
To put the network to training use the command below. Hyperparameters can be changed by giving inputs via terminal. Please go through the file train.py for knowing about hyperparameters


`python3 train.py`





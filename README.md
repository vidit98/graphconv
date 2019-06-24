## Introduction

![Semantic Segmentation](https://ibb.co/8bmbmxM)


Semantic segmentation is the task of assigning each pixel a class. Many different methods are proposed for the same. This a pytorch implementation of [this](https://papers.nips.cc/paper/8135-beyond-grids-learning-graph-representations-for-visual-recognition.pdf) paper. 
## Highlights

* Model can be distributed over many GPUs: pytorch provides the facility to divide data into various GPUs and train the network parallely. But here you can divide the model in various GPUs.
* Live Data Visualization: You can visualize the training curve live using visdom.
* Flexible to remove and add GCU units.

# EM509 Stochastic Processes Project : Random Walks for image segmentation

## Abstract

First random walks are introduced in the order of 1D (without barriers and with barriers), 2D (without barriers and with barriers). Then Markov property is explained. The Markov property is proved for the 1D simple case and a complicated 2D case with both absorbing and reflecting barriers. Image segmentation problem is introduced from an application point of view and converted into a mathematical formulation. Random walker algorithm for image segmentation is introduced and it is proven to be a Markov process. The study is concluded by implementing the Random Walker algorithm and testing it by segmenting a set of images.

## Documentation

[PDF](report.pdf), [Latex Code](/report/)

## Code

[segment.py](segment.py)

## Examples

### Objective : image segmentation
Input and the desired output of a perfect image segmentation algorithm

![Input and the desired output of a perfect image segmentation algorithm](/examples/image-segmentation.png)

### Results : image segmentation using random walks

![Input and the output of image segmentation using random walks](/examples/efac-all.jpg)

![Input and the output of image segmentation using random walks](/examples/fish-all.jpg)



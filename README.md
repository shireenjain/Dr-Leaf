# Dr. Leaf - Automated Plant Disease Detection and Prevention
This project aims to build a robust Machine Learning system for automatic plant disease detection. Additionally a user-interface has also been developed to provide information related to plant diseases and to suggest supplements.
[Kaggle Notebook](https://www.kaggle.com/code/jagritpant21bsa10015/dr-leaf?scriptVersionId=176499358)

## Dataset used:
* consists of 87K rgb images of healthy and diseased crop leaves 
* categorized into 38 different classes acording to fungal, bacterial, or viral diseases etc.
* split into a ratio of 80:20 for training and validation
* Plants (14): Tomato, Grape, Orange, Soybean, Squash, Potato, Corn, Strawberry, Peach, Apple, Blueberry, Cherry, Pepper_bell, Raspberry

## ResNet Architecture:
* Layers: The architecture consists of several convolutional layers organized into stages:
* conv1: Applies a ConvBlock with 64 output channels.
* conv2: Applies another ConvBlock with 128 output channels, followed by a pooling layer that reduces the data size.
* res1: Uses two ConvBlocks with 128 output channels each. 
* Repeated Stages: This pattern (ConvBlock -> Pooling -> Residual Block) is repeated for stages 3 and 4, with increasing numbers of output channels (256 and 512 respectively).

### Accuracy: 99.2%


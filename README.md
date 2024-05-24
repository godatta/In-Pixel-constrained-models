# Pixel_contrained_net_internal


This repository supports the training and testing of a MobileNetV2 model with the first convolutional layer being implemented inside the sensor. This requires the incorporation of a custom convolution layer that is implemented inside MobileNetv2.py (look for the 'customConv2' class).

First download the visual wake words dataset from https://github.com/Mxbonn/visualwakewords?tab=readme-ov-file. Then, please change the root and annotation paths in lines 126-127 and 129-130 according to your downloaded data.

To test the in-pixel constrained models, please download our model checkpoint from xx and place the same in the home directory

Then, please run run_vww2.py

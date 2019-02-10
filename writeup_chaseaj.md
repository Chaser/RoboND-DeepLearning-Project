# Project: Follow Me

## Engineer: Chase Johnson

---
[//]: # (Image References)

[image1]: ./images/following.png
[image2]: ./images/object_detection_types.jpeg
[image3]: ./images/fcn_structure.png


**Aim:**  The aim of the `Follow Me` project is to identify and track a target. Identification and tracking is achieved by training a deep neural network. The target is a person called a "hero" which will be mixed in with other people (image below)

![alt text][image1]

This particular application is at the forefront of robotics and computer vision as it can be used in a number of applications such as autonomous vehicles.

# Deep Learning Architecture
Being able to detect objects or perceive has become an important task across a variety of industries. Depending on the perception tasks, for instance asking the question "Is an object present" is different to "How many objects are present". For this reason visual perception is divided into four distinct categories.

- Classification, assigns a label to an entire image
- Localization, assigns a bounding box to a particular label
- Object Detection, draws multiple bounding boxes in an image
- Image segmentation, creates precise segments of where objects lie in an image

![alt text][image2]

Categories and Image provided by reference [1]

For this particular project, the question taht is asked is "Where is an object", therefore require a deep learning model that is able to detect an object within a video. As the "hero" (target) is mixed among an environment including other people we need to leverage `semantic segmentation` as it achives fine-grained interfence by labelling each pixel in relation to its features and location to other pixels in an image.

A model that is able to achieve `semantic segmentation` is `Fully Convolution Networks` (FCNs). FCNs are an extension of Convolution Neural Networks (CNN), where instead of having a fully connected layer they create an encoder/decoder topology. This inherits the benfits of CNNs but also preserves spatial information allow allowing us to be able to detect **if** our hero is in the scene and if so **where**.

## Fully Covolution Networks (FCNs)

![alt text][image3]

### Encoder

### 1x1 Convolution Layer

### Decoder

### Skip Connections

## FCN Model

# Hyperparameters


# Training


# Performance


# Final Result and Improvements.

# Resources
[1] [How to easily Detect Objects with Deep Learning on Raspberry Pi](https://medium.com/nanonets/how-to-easily-detect-objects-with-deep-learning-on-raspberrypi-225f29635c74?fbclid=IwAR2eEoHgWOsdErlzY4HOvmeord_5gw-0q4O8BWUHR-R_LIVrNmubfVWzXmQ)
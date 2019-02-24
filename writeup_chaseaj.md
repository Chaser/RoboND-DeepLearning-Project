# Project: Follow Me

## Engineer: Chase Johnson

---
[//]: # (Image References)

[image1]: ./images/following.png
[image2]: ./images/object_detection_types.jpeg
[image3]: ./images/fcn_structure.png
[image4]: ./images/covnets.png
[image5]: ./images/convolutions.png
[image6]: ./images/fcn_encoder.png
[image7]: ./images/cnn_structure.png
[image8]: ./images/1x1_convolution_layer.png
[image9]: ./images/fcn_decoder.png
[image10]: ./images/bilinear.png
[image11]: ./images/fcn_skip_connections.png
[image12]: ./images/fcn-model.jpg

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
As indicated above `Fully Convolution Networks` or FCNs leverage Convolution Neural Networks (CNN), where instead of having a fully connected layer a 1x1 convolution layer and a decoder are part of the model (image below).

![alt text][image3]

FCNs take advantage of three special techniques
1) Replace fully connected with 1x1 convolution layer
2) Upsampling through the use of transposed convolution layers
3) Skip Connections

These techniques will be discussed in detail in the following sections.

### Encoder
A fully connected neural network has the abiliy to learn features as well as classify data, however its not practicical to apply to images. The reason being that a high number of neurons required. Convolution Neural Networks is able to reduce the parameters by leveraging weight sharing. Weight sharing is an important optimisation as often in perception problems the location of object is not important, this is known as statistical invariance.

For instance in the image below, if the question is "Is there a kitten" then this problem is considered classification, so we don't care where in the image the kiten exists. Analysis of where is costly in that it requires more weights or parameters and ultimately costs in training and detection time.

![alt text][image4]

CNNs are able to reduce paramters by applying a filter and scanning over the input layer (image below). 

![alt text][image5]

Its common to have more than one filter as different filters pick up up different qualities of a patch. The amount of filters per convolution layer is called a filter depth `k`. The filter depth translates to the amount of neurons each patch (kernel) is connects to.

As with deep neural network topology this process is repeated multiple times  which ends up forming a pyramid (image below). This provides a structure that reduces the spacial dimenions and the weights required while improving understanding by increasing the filter depth.

![alt text][image6]

### 1x1 Convolution Layer
A CNN final layer is a fully connected layer (image below). 

![alt text][image7]

However the problem with a fully connected layer is that data is flattened losing the spatrial data within the input. While useful for classification requirements it's not useful for detection and segmentation.

This is overcome by replacing the fully connected layer by using a 1x1 convolution layer with a kernel and stride of 1. By using a 1x1 convolution layer, the network is able to retain spatial information from the encoder along with the benfit that it works on any input size.

![alt text][image8]

### Decoder
The decoder connects to the 1x1 convolution layer where the decoder can compose of bilinear upsampling layers or transposed convolution layers.

![alt text][image9]

Bilinear upsampling is a resampling technique that utilizes the weighted average of four nearest known pixels, located diagonally to a given pixel, to estimate a new pixel intensity value. The weighted average is usually distance dependent.

![alt text][image10]

Transposed Convolutions reverse regular convolution layers by upsampling the previous layer to a desired resolution or dimension. The process involves multiplying each pixel of your input with a kernel or filter.

Bilinear upsampling process is less computation intense however pays the price in terms of lose of details compared to transposed convoultions. In this project we will use bilinear upsampling.

### Skip Connections
When performing convulations, spatial information is lost to save or weights and calculation time. With FCNs a method called skip connections can be utilized which allows for information to be retained. This is achieved by "wiring" the output of encoder layers to combine with decoder layers (image below).

![alt text][image11]

## FCN Model
Based on the components above a model was constructured consisting of two encoders, 1x1 convolutional layer and two decoders. From the decoder the last process is a `softmax` to ensure normalizations of the probability distribution.

To ensure that spatial information is retained skip connections were also used between `encoder0` and `decoder0` and the `inputs` and `decoder1`. The architecture and code are below.

![alt text][image12]

```python
def fcn_model(inputs, num_classes):
    
    # Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    f = 32
    encoder0 = (encoder_block(inputs, f, 2))               # 32
    encoder1 = (encoder_block(encoder0, f*2, 2))           # 64
    # Add 1x1 Convolution layer using conv2d_batchnorm().  
    oneconv = conv2d_batchnorm(encoder1, f*2*2, 1, 1)      # 128
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder0 = decoder_block(oneconv, encoder0, f*2)       # 64
    decoder1 = (decoder_block(decoder0, inputs, f))        # 32
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(decoder1)
```

# Hyperparameters


# Training

# Performance


# Final Result and Improvements.

# Resources
[1] [How to easily Detect Objects with Deep Learning on Raspberry Pi](https://medium.com/nanonets/how-to-easily-detect-objects-with-deep-learning-on-raspberrypi-225f29635c74?fbclid=IwAR2eEoHgWOsdErlzY4HOvmeord_5gw-0q4O8BWUHR-R_LIVrNmubfVWzXmQ)
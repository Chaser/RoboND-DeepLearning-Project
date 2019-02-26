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
[image13]: ./images/hyperparameter-batch-size-1.png
[image14]: ./images/hyperparameter-batch-size-10.png
[image15]: ./images/hyperparameter-batch-size-20.png
[image16]: ./images/hyperparameter-batch-size-32.png
[image17]: ./images/hyperparameter-batch-size-64.png
[image18]: ./images/hyperparameter-batch-size-100.png
[image19]: ./images/hyperparameter-lr-0.01.png
[image20]: ./images/hyperparameter-lr-0.005.png
[image21]: ./images/hyperparameter-lr-0.001.png
[image22]: ./images/hyperparameter-final.png


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
There were a number of hyperparamers that effect the performance of the FCN model. These were: 

* `batch_size`: number of training samples/images that get propagated through the network in a single pass.
* `num_epochs`: number of times the entire training dataset gets propagated through the network.
* `steps_per_epoch`: number of batches of training images that go through the network in 1 epoch. 
* `validation_steps`: number of batches of validation images that go through the network in 1 epoch. This is similar to steps_per_epoch, except validation_steps is for the validation dataset. 
* `workers`:` maximum number of processes to spin up. This can affect your training speed and is dependent on your hardware. We have provided a recommended value to work with.

## Baseline
The following settings were used as baseline to tune the hyperparameters:

```python
learning_rate = 0.01
batch_size = 1
num_epochs = 10
steps_per_epoch = 200   # 4131 imgs. 4131 / batch_size
validation_steps = 50  # 1184 imgs. 1184 / batch_size
workers = 100
```

With the baseline hyperparmeters the following performance was achieved.

![][image13]

*learning_rate=0.01, batch_size=1, num_epochs=10, steps_per_epoch=200, validation_steps=50*

```
loss: 0.0622
val_loss: 0.0562
final_IoU = 0.00624209070893
final_score = 0.000980899968546
```

## Batchsize
Mini-batching is a technique for training on subsets of the dataset instead of all the data at one time. The `batch_size` was increased to 10, then 20, 32, 64 and finally 100. 

![][image14]

*learning_rate=0.01, batch_size=10, num_epochs=10, steps_per_epoch=420, validation_steps=120*

```
loss: 0.0361
val_loss: 0.0431
final_IoU = 0.514525885561
final_score = 0.364092469867
```

![][image15]

*learning_rate=0.01, batch_size=20, num_epochs=10, steps_per_epoch=210, validation_steps=60*

```
loss: 0.0380
val_loss: 0.0521
final_IoU = 0.324593859237
final_score = 0.226890022008
```

![][image16]

*learning_rate=0.01, batch_size=32, num_epochs=10, steps_per_epoch=130, validation_steps=40*

```
loss: 0.0403
val_loss: 0.0489
final_IoU = 0.500249301806
final_score = 0.340426824573
```

![][image17]

*learning_rate=0.01, batch_size=64, num_epochs=10, steps_per_epoch=65, validation_steps=20*

```
loss: 0.0371
val_loss: 0.0637
final_IoU = 0.229447864258
final_score = 0.161728876543
```

![][image18]

*learning_rate=0.01, batch_size=100, num_epochs=10, steps_per_epoch=42, validation_steps=12*

```
loss: 0.0291
val_loss: 0.0475
final_IoU = 0.517982925573
final_score = 0.370449327229
```

## Learning Rate
The learning rate is related to Stochastic Gradient Decent (SGD) and how much the weights of network are adjusted with respect to the loss gradient. If the value is too small then gradient descent can be slow, too large it may overshoot the minimum or fail to converge.

The following settings were used to examine the `learning_rate` hyperparameter.

```python
learning_rate = 0.01, 0.005, 0.001
batch_size = 32
num_epochs = 10
steps_per_epoch = 200       # 4131 imgs. 4131 / batch_size
validation_steps = 50       # 1184 imgs. 1184 / batch_size
workers = 100
```

![][image19]

*learning_rate=0.01, batch_size=32, num_epochs=10, steps_per_epoch=200, validation_steps=50*

```
loss: 0.0333
val_loss: 0.0443
final_IoU = 0.474703688247
final_score = 0.345613086135
```

![][image20]

*learning_rate=0.005, batch_size=32, num_epochs=10, steps_per_epoch=200, validation_steps=50*

```
loss: 0.0268
val_loss: 0.0466
final_IoU = 0.575321307718
final_score = 0.406483295804
```

![][image21]

*learning_rate=0.001, batch_size=32, num_epochs=10, steps_per_epoch=200, validation_steps=50*

```
loss: 0.0216
val_loss: 0.0285
final_IoU = 0.536894087208
final_score = 0.40077186488
```

## Epochs
An epoch is a single forward and backward pass of the whole dataset. Increasing the number of epochs increases the models awareness of the dataset. 

Epoch tuning was approached differently to `batch_size` and `learning_rate`. In this case the epoch value was set to 50 and run to see if improvements could be witnessed.

```python
learning_rate = 0.001
batch_size = 32
num_epochs = 50
steps_per_epoch = 200   # 4131 imgs. 4131 / batch_size
validation_steps = 50  # 1184 imgs. 1184 / batch_size
workers = 100
```

It was clear after the model had been running for a while that increasing the epoch beyond 30 was over-fitting the training data.

```
Epoch 1/100 -  loss: 0.0208 - val_loss: 0.0285
Epoch 10/100 - loss: 0.0208 - val_loss: 0.0322
Epoch 15/100 - loss: 0.0202 - val_loss: 0.0316
Epoch 20/100 - loss: 0.0201 - val_loss: 0.0343
Epoch 25/100 - loss: 0.0196 - val_loss: 0.0330
Epoch 30/100 - loss: 0.0193 - val_loss: 0.0361
Epoch 35/100 - loss: 0.0196 - val_loss: 0.0328
Epoch 40/100 - loss: 0.0196 - val_loss: 0.0287
Epoch 45/100 - loss: 0.0191 - val_loss: 0.0311
Epoch 50/100 - loss: 0.0191 - val_loss: 0.0331
Epoch 55/100 - loss: 0.0198 - val_loss: 0.0283
Epoch 60/100 - loss: 0.0193 - val_loss: 0.0284
Epoch 65/100 - loss: 0.0183 - val_loss: 0.0304
Epoch 70/100 - loss: 0.0190 - val_loss: 0.0337
Epoch 75/100 - loss: 0.0189 - val_loss: 0.0292
Epoch 80/100 - loss: 0.0187 - val_loss: 0.0285
Epoch 85/100 - loss: 0.0179 - val_loss: 0.0294
Epoch 90/100 - loss: 0.0180 - val_loss: 0.0276
Epoch 95/100 - loss: 0.0182 - val_loss: 0.0347
Epoch 100/100 - loss: 0.0181 - val_loss: 0.0298
```

![alt text][image22]

```
final_IoU = 0.568096222338
final_score = 0.412785092946
```

# Result
The final score for the model was 41.2% with the following results:

```python
learning_rate = 0.001
batch_size = 32
num_epochs = 50
steps_per_epoch = 200   # 4131 imgs. 4131 / batch_size
validation_steps = 50  # 1184 imgs. 1184 / batch_size
workers = 100
```

```python
# Scores for while the quad is following behind the target. 
number of validation samples intersection over the union evaulated on 542
average intersection over union for background is 0.9949233920264684
average intersection over union for other people is 0.31915892396839296
average intersection over union for the hero is 0.8927244381945009
number true positives: 539, number false positives: 1, number false negatives: 0
```
```python
# Scores for images while the quad is on patrol and the target is not visable
number of validation samples intersection over the union evaulated on 270
average intersection over union for background is 0.9831018147993
average intersection over union for other people is 0.6469095769858776
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 85, number false negatives: 0
```

```python
# This score measures how well the neural network can detect the target from far away
number of validation samples intersection over the union evaulated on 322
average intersection over union for background is 0.9960946109154941
average intersection over union for other people is 0.4231563270849253
average intersection over union for the hero is 0.2413463362618471
number true positives: 152, number false positives: 7, number false negatives: 149
```

```
final_IoU = 0.567035387228
final_score = 0.419958684432
```

# Adaptability for Following Other Objects
It was asked if "the model and data would work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required"

To answer the first component of the question is no, the **model** and **data** at present would not work for **following** another object. Semantic Segmentation leverages the proximity of certain pixels and in this case the model was trained to label three features 1: Hero, 2: Non-Hero and 3: other items (background). To be able to follow another object new data would have to be aquired and the model be taught these labels. 

Once taught the question of the quality of the model can be considered for instance would the model be able to determine the difference between a dog and a car? I would say yes, however the question of a can it determine if an object is a dog or a cat? I would say no. The network is not very deep and I believe would cause issues.

# Future Improvements.
The IoU result of 41.9% while sufficient for the project can be signficantly improved. First improvement would be to increase the training data set. The second improvement could a decrease in the learning rate while increasing the number of epochs. This would provide a higher degree of loss accuracy however would increase the training time significantly.

The main area of improvement is to use a "deeper" model. The current network with two encoder/decoder layers its clear is quite small compared to a network such as ResNet which has 100 layers.


# Resources
[1] [How to easily Detect Objects with Deep Learning on Raspberry Pi](https://medium.com/nanonets/how-to-easily-detect-objects-with-deep-learning-on-raspberrypi-225f29635c74?fbclid=IwAR2eEoHgWOsdErlzY4HOvmeord_5gw-0q4O8BWUHR-R_LIVrNmubfVWzXmQ)
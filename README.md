# Project: Follow Me (Deep Learning)
### Submission for Udacity Robotics Software Engineer Nanodegree Program
### Sebastian Castro - 2018

[//]: # (Image References)

[intro_img]: ./docs/images/intro_image.PNG
[data_col_1]: ./docs/images/data_collection_setup_1.PNG
[data_col_2]: ./docs/images/data_collection_setup_2.PNG
[network_arch]: ./docs/images/network_arch.PNG
[train_plots]: ./docs/images/training_plots.PNG
[results_follow]: ./docs/images/results_follow.PNG
[results_notarget]: ./docs/images/results_notarget.PNG
[results_far]: ./docs/images/results_far.PNG

---

## Introduction
In this project, we implemented a person detector for a quadcopter to identify and follow a "hero", 
or person of interest, in a simulated environment.

[Project introduction][intro_img]

This was achieved using a fully convolutional network (FCN) to perform semantic segmentation -- 
that is, classifying each pixel of an input image into one of 3 classes:

* Background
* Person
* Hero

## Methodology

### Data Collection and Preprocessing
To improve training results, the training and validation data provided was augmented with 
more data collected from patrolling and following the target.

Some keys to getting good data were:
* Ensuring several non-hero persons were around the hero, by creating many spawn points around the hero path
* Patrol points at varying altitudes, so long as there were some visible people
* Tried to use diverse backgrounds (grass, road, footpath), including obstructions such as rocks, colorful trees, etc.

![Data collection setup 1][data_col_1] 
![Data collection setup 2][data_col_1] 

### Neural Network Architecture
The network architecture selected was inspired by [U-Net](https://arxiv.org/abs/1505.04597), 
which halves the image size and doubles the feature depth at each step. Unlike the architecture 
in the paper, which has 4 encoder and decoder layers with passthrough, our network only has 3. 
The 1x1 convolution layer at the middle layer of the neural network therefore has 512 filters 
instead of 1024.

![Network architecture diagram][network arch] 

We found that the network did not need any more layers because the middle layer already is of size 
20-by-20 pixels. The 1x1 convolution was used to expand (double, in this case) the number of features 
in a way that uses less parameters/weights than a fully-connected layer. In addition, the use of 
1x1 convolutions helps retain spatial information in the image.

The nonlinearity in this system is caused by the ReLU activation function at the end of each encoder 
and decoder block. This is defined in the `separable_conv2d_batchnorm` function used in defining all 
these layers.

```
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer
```

Also note all features either double or half in size (160 > 80 > 40 > 20 > 40 > 80 > 160). 
This is the case because all encoder filters use a stride of 2, and all decoder filters similarly 
upsample the image by a factor of 2 to recover the final output image size.

```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([upsampled_layer, large_ip_layer],axis=3)
    
    # Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concat_layer, filters, 1)

    return output_layer
```

In accordance with the diagram, the function that defines the entire network is as follows. 
Notice that we use skip connections (directly connecting encoders and decoders), so the network 
can use data at multiple resolutions to improve training results.

```
def fcn_model(inputs, num_classes):
    
    # Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encoded_layer_1 = encoder_block(inputs, 64, (2,2))
    
    encoded_layer_2 = encoder_block(encoded_layer_1, 128, (2,2))
    encoded_layer_3 = encoder_block(encoded_layer_2, 256, (2,2))
    
    # Add 1x1 Convolution layer using conv2d_batchnorm().
    mid_layer = conv2d_batchnorm(encoded_layer_3, 512, 1, 1)
    
    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoded_layer_3 = decoder_block(mid_layer, encoded_layer_2, 256)
    decoded_layer_2 = decoder_block(decoded_layer_3, encoded_layer_1, 128)
    decoded_layer_1 = decoder_block(decoded_layer_2, inputs, 64)

    # The function returns the output layer of your model. "decoded_layer_1" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decoded_layer_1)
```

### Training Hyperparameters
Learning rate was **0.005**. We chose a learning rate that was small enough to cause a constant 
decrease in loss. As a general rule, a larger learning rate can lead to a loss minimum faster, but 
the final loss may be higher than if a smaller learning rate were used. By experimenting with 
learning rates, we found that 
* Learning rates of 0.01 and 0.005 yielded similar results, with 0.005 leading to slightly 
lower loss. 
* A smaller learning rate of 0.0025 did not yield better results than 0.005
So, 0.005 was selected as our final value.

Batch size was set to **100**, since a batch size of 128 or higher caused out of memory issues 
on the EC2 instance given our network parameters. Specifically, a value of 100 made the 
calculations of selecting number of epochs and validation steps easy to compute.

Since the training set had **5672** images and the validation set **1841** images, the number of training 
and validation steps per epoch were set to **60** and **20**, respectively. This comes out to be slightly 
higher than number of images divided by batch size, to ensure that each training and validation 
image is used during at least once during any given epoch.

We found that training the network for **15-20 epochs** was adequate. As an experiment, we set 
the number of epochs to 50. Notice that after about 18 epochs, training loss does not decrease much 
further and validation loss begins oscillating.

[Training progress plots][train_plots]

## Results
The final trained network weights can be found in `data/weights/model_weights/`.

Our final network results were:
* Final IoU of **54.75%**
* Score weight of **74.6%** 
* Overall grade score of **40.86%** 

We found that the network worked well for detecting the hero and distinguishing it from regular 
persons up close, but did not detect persons well from far away. See the images below as an example.

We can also confirm this with the following metrics from the training code:

* No false positives or false negatives when following the target 
* 30 false positives and no false negatives when the quad is on patrol and target is not visible
* 1 false positive and **190 false negatives** (compare to 111 true positives) when trying to track the target from far away

### Following the Target
[Sample results when following the target][results_follow]

### Target is Not Visible
[Sample results when target is not visible][results_notarget]

### Detecting from Far Away
[Sample results from far away][results_far]

## Future Enhancements
Since the proposed network barely meets the passing requirements for the project, there are several 
factors to consider in the future to improve results. 

### Collecting More Data
As discussed in the results section, the biggest weakness of the network had to do with detecting 
targets from far away. Therefore, given more time we could try to collect better patrol data at 
different altitudes that ensures the hero and other non-hero persons are visible.

From a data collection standpoint, getting good data that meets these criteria was far more difficult 
than getting up-close data, since the follow mode in the simulator was quite robust, whereas data 
collection from far away relied heavily on carefully chosen spawn points, hero paths, and patrol points.

### Network/Training Parameters
Our network architecture seemed to be test since the training and validation loss did not decrease much 
past 15-20 training epochs, and slightly changing the learning rate did not affect this observation.

A deeper network with more training parameters may have been better at fitting to the training data, but 
could also be prone to overfitting. 

Ultimately, the quality of data appeared to be more important than the network architecture 
for this particular problem.

### Further Discussion
This network might extend to detect other types of objects, though it may not work as well. 
Minimally, we would need to at least change the output layer definition to account for the number of 
classes in our new classification problem. This is because the 3 classes in the output layer are 
specifically background, hero, and non-hero person. 

The problem in this project was made slightly easier than real-life since there are sharp differences in 
color between the backgrounds (typically green/gray) and the persons (bright red for the hero, and 
other non-background colors like blue/white for regular persons). Taking advantage of color, the 
network might adequately find contrasting foreground objects such as dogs or cats even if their shapes 
are different from people. However, this would probably only help at telling you whether something is 
background or non-background, and not necessarily dog vs. cat (unless our simulated cats, for example, 
happened to be just as red as the hero, while dogs used other colors).

One way this network could be partially used in other detection problems is through transfer learning. 
Perhaps some of the learned features in the encoding phase can identify rough features such as limbs,
distinct color blobs, etc. If we take this same network architecture, we could potentially retrain it
on new data without having to retrain the entire network.
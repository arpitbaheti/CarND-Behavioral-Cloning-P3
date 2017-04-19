**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

**Rubric Points**
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
**1.Files Submitted & Code Quality**

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

**2. Submission includes functional code**
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

**3. Submission code is usable and readable**

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

**4.Model Architecture and Training Strategy**

1. Solution Design Approach
The Nvidia model was adopted for training, because it gave better result after experimenting with other kinds of model (e.g. ). The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. Converse to the Nvidia model, input image was split to HSV planes before been passed to the network.

Image was normalized in the first layer. According to the Nvidia paper, this enables normalization also to be accelerated via GPU processing.

Convolution were used in the first three layers with 2x2 strides and a 5x5 kernel, and non-strided convolution with 3x3 kernel size in the last two convolutional layers.

The convolutional layers were followed by three fully connected layers which then outputs the steering angle.

2. Final Model Architecture

| Layer (type)                   | Output Shape       |  Param #  |  Connected to             |
|:------------------------------:|:------------------:|:---------:|:-------------------------:|
| lambda_1 (Lambda)              |  (None, 66, 200, 3)|   0       |    lambda_input_1[0]0]    |
|                                |                    |           |                           |
| convolution2d_1 (Convolution2D)|  (None, 31, 98, 24)|    1824   |    lambda_1[0]0]          |
|                                |                    |           |                           |
| convolution2d_2 (Convolution2D)|  (None, 14, 47, 36)|   21636   |    convolution2d_1[0]0]   |
|                                |                    |           |                           |
| convolution2d_3 (Convolution2D)|  (None, 5, 22, 48) |   43248   |    convolution2d_2[0]0]   |
|                                |                    |           |                           |
| convolution2d_4 (Convolution2D)|  (None, 3, 20, 64) |  27712    |   convolution2d_3[0]0]    |
|                                |                    |           |                           |
| convolution2d_5 (Convolution2D)|  (None, 1, 18, 64) |   36928   |    convolution2d_4[0]0]   |
|                                |                    |           |                           |
| flatten_1 (Flatten)            |  (None, 1152)      |   0       |    convolution2d_5[0]0]   |
|                                |                    |           |                           |
| dense_1 (Dense)                |  (None, 100)       |   115300  |    flatten_1[0]0]         |
|                                |                    |           |                           |
| dropout_1 (Dropout)            |  (None, 100)       |   0       |    dense_1[0]0]           |
|                                |                    |           |                           |
| dense_2 (Dense)                |  (None, 50)        |   5050    |    dropout_1[0]0]         |
|                                |                    |           |                           |
| dropout_2 (Dropout)            |  (None, 50)        |   0       |    dense_2[0]0]           |
|                                |                    |           |                           |
| dense_3 (Dense)                |  (None, 10)        |   510     |    dropout_2[0]0]         |
|                                |                    |           |                           |
| dense_4 (Dense)                |  (None, 1)         |   11      |    dense_3[0]0]           |

Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0

3. Model Visualization

![alt text][image1]

4. Attempts to reduce overfitting in the model**

Overfitting was reduced by using Dropout (0.2) between first two layer.
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

5. Model parameter tuning**
An Adam optimizer was used for optimization. This requires little or no tunning as the learning rate (0.0001) is adaptive. In addition, checkpoint and early stop mechanisms were used during training to chose best training model by monitoring the validation loss and stopping the training if the loss does not reduce in three consecutive epochs.

The model includes RELU layers to introduce nonlinearity.

**3. Creation of the Training Set & Training Process**

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road .

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would increase the dataset and remove the bias of left turn on the track, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 42000 number of data points. I then preprocessed this data by 
1. Normalization
2. Flipping images Randomly
3. Since the original image size is 160x320 pixels, we crop image 70 pixel from top and 25 pixel from bottom to obtain the region of interest.
4. Take image from all the camera with the correction factor of 0.2
5. We simulate different brightness occasions by converting image to HSV channel and randomly scaling the V channel.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 to chose best training model by monitoring the validation loss and stopping the training if the loss does not reduce in three consecutive epochs.

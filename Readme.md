# **Behavioral Cloning Project 3** 
## Udacity Self Driving Car Nanodegree

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* bs512-lr5e-05-do0.0-epoch079-loss0.000739-os0.0-model.h5 containing a trained convolution neural network
* balanced_csv_gen.py for creating a balanced dataset for training
* image_visualize.py for ran choosing an image from the set and predicting the steering angle
* Readme.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py bs512-lr5e-05-do0.0-epoch079-loss0.000739-os0.0-model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is similar to the model described in NVIDIA's paper "End to End Learning for Self Driving Cars" https://arxiv.org/pdf/1604.07316v1.pdf


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers but the final model didn't use dropout. It was tested with varying dropout rates and the best validation loss was used. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The design approach went as follows: Collect a large amount of data, balance the data to be used, process the images to increase effectiveness and dataset size, run through the training pipeline.

First a large amount of data was collected from driving the track manually. About ten laps in each direction at first and then a couple laps of recovery driving as well. This gave a large dataset to begin working with. I then created a script to generate a histogram of the images based on steering angle of the training data. (balanced_csv_gen.py) This allowed me to visualize how balanced the data was, and later correct it. Original data histogram:

![alt text](https://github.com/NickGoumas/CarND-BehavioralCloning-P3/blob/master/images/original_histogram.png?raw=true "Original Histogram")

The histogram only plots the angles absolute value because I know the images will be flipped in the model pipeline anyway. It's obvious a model trained on this data could easily favor driving strait. What we want is a roughly equal spread of steering angles so the images are the dominating factor, not the number of images of a certain angle. Looking at the balanced_csv_gen.py script will show how I iterated through the large dataset and constructed a smaller but more balanced "driving_log_balanced.csv" file. The histogram for this file: 

![alt text](https://github.com/NickGoumas/CarND-BehavioralCloning-P3/blob/master/images/balanced_histogram.png?raw=true "Balanced Histogram")

The total frames represented in each graph is listed in the title. The balanced histogram lost more than two thirds of the data but is now much more balanced. This proved to help the model greatly. It also allowed for faster training as opposed to running the whole dataset through the pipeline.

Once the dataset was ready the model pipeline reads the images and performs some operations to increase the abilities of the model. First the pixel values are normalized, then the images are cropped to reduce uninteresting data. Below are a couple images that that show the training and predicted angles of random images as well as the cropped area. The blue bars on top and bottom of the images are removed before being passed to the model. 

![alt text](https://github.com/NickGoumas/CarND-BehavioralCloning-P3/blob/master/images/good_prediction_fig.png?raw=true "Good Prediction")
![alt text](https://github.com/NickGoumas/CarND-BehavioralCloning-P3/blob/master/images/bad_prediction_fig.png?raw=true "Bad Prediction")

In the model pipeline the images are also flipped horizontally and the steering angle's sign is switched. This results in doubling the amount of training data.


#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

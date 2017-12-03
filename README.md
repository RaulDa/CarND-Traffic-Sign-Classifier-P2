# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/sign_count.jpg  "Visualization"
[image2]: ./results/color_gray.png "Grayscaling"
[image4]: ./results/webImages_grayscale.png "Web images"
[image14]: ./results/conv1.png "First convolution"
[image15]: ./results/conv2.png "Second convolution"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/RaulDa/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Data set summary

I used the NumPy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 examples
* The size of the validation set is 4410 examples
* The size of test set is 12630 examples
* The shape of a traffic sign image is 32x32 pixels with RGB scale (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many images corresponding to each traffic sign the training set contains:

![alt text][image1]

As shown in the figure, the data set contains an unequal number of images for each traffic sign. The sign with the most images is the speed limit of 50 km/h (label 2), with around 2000 samples. However, the speed limit of 20 km/h (label 0) has around 200 samples. This will determine the probability of success for each sign and also the overall accuracy of the network for the data set.

### Design and Test a Model Architecture

#### 1. Data preprocessing

As a first step, I decided to convert the images to grayscale. Since color is not a factor determining the success of the classification, omiting this information is suppossed to increase the performance of the network. In fact, the LeCun paper shows that implementing grayscale the network is more accurate.

Here is an example of a traffic sign image before and after grayscaling:

![alt text][image2]

As a last step, I normalized the image data, in order to get a representation with zero mean and an equal variance. This features will do that the optimizer works better when estimating and reducing the error.


#### 2. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale and normalized image   							|
|						|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x8 				|
|						|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x20      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x20 				|
|						|												|
| Fully connected		| Inputs 500, outputs 180        									|
| RELU		|        									|
| Dropout				| Keep probability 0.5        									|
|						|												|
| Fully connected		| Inputs 180, outputs 140        									|
| RELU		|        									|
| Dropout				| Keep probability 0.5        									|
|						|												|
|	Fully connected		|	Inputs 140, outputs 43											|

I decided to increase the feature maps of both convolutions with regard to the original LeNet. For the first layer the number is 8 and for the second layer 20 feature maps. Taking as assumption that 6 and 14 worked well for the MNIST data set, a configuration with more feature maps should work better for the sign images, since their features are more complex.

Finally I added a regularization (dropout) in the layers 3 and 4, with a keep probability of 0.5.


#### 3. Model training

To train the model, I used the following parameters:

| Parameter         		|     Value	        					|
|:---------------------:|:---------------------------------------------:|
| Epochs         		| 40   							|
|	Batch size					|	64											|
| Learning rate 	| 0.00075 	|

Since it was chosen to increase the number of epochs until the accuracy stops growing, I also decided to decrease the learning rate, so that it learns slower but better, above all for the last epochs. Additionally, a decrease in the batch size of 64 provided slightly better results for the current network configuration.

Finally, the weights were initialized with zero mean and equal variance. This configuration is needed to make the Adam optimizer (method chosen by the initial LeNet architecture) work properly.

#### 4. Solution approach and results

As previously mentioned, LeNet was taken as base for building the network architecture. The reason is that this architecture provides good results for the MNIST data set, which is of similar complexity in comparison with the traffic sign data set (the last one is slightly more complex).

Some changes were applied to the initial LeNet. The feature maps were increased from 6 and 14 to 8 and 20, according to the complexity increase of the new data set (includes more features to be detected). Also, the dropout regularization technique was applied to the third and fourth layers. The reason is that this method has been recently proved to reduce successfully the overfitting problem.

With this changes and also the hyperparameter selection explained previously, I expected a significant increase of the accuracy. The initial validation accuracy was 0.89, and with the mentioned configuration the results are the following:

* training set accuracy of 0.988
* validation set accuracy of 0.970
* test set accuracy of 0.958

The slight difference in the accuracy of both validation and test sets shows that that the selected parameters and configuration prevent overfitting, that is, the network learn to detect features instead of just "learn" only the training set. Of course, the network provide a significant increase in the accuracy, always above 0.93.


### Test a Model on New Images

#### 1. Traffic signs found on the web

Here are ten German traffic signs that I found on the web, after grayscaling and normalizing:

![alt text][image4]

The whole images can be found on the [web_images](https://github.com/RaulDa/CarND-Traffic-Sign-Classifier-Project/tree/master/web_images) folder of the repository.

The first image might be difficult to classify because it looks slightly blurred. Also the stop image due to the amount of features to detect and the slippery road and children crossing images, for containing detailed and small features that could me more difficult to detect by the network.

#### 2. Predictions of new traffic signs. Accuracy comparison with validation and test results

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (50 km/h)     		| No passing for vehicles over 3.5 metric tons   									|
| No passing     			| No passing 										|
| Priority road					| Priority road											|
| Yield	      		| Yield					 				|
| Stop			| Stop      							|
| Slippery road      		| Slippery road   									|
| Children crossing     			| Road narrows on the right 										|
| Wild animal crossing					| Wild animal crossing											|
| Go straight or right	      		| Go straight or right					 				|
| End of no passing			| End of no passing     							|


The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. This compares reasonably to the accuracy on the test set of 95.8%. The accuracy is lower but high though, and a lower result is expected since the number of images is also lower.

Although the speed limit of 50 km/h signal is the one with the most samples in the data set, it is not detected correctly. I believe the reason is that the image found on the web is slightly blurred. The children crossing signal is also not detected correctly. In my opinion, the fact that the features are more difficult to detect in comparison with other signs, and the number of samples in the data set (around 500) being not one of the highest, make that the detection fails.

#### 3. Predictions certainty through softmax probabilities

The 5 highest softmax probabilities for the signs Speed limit (50 km/h), priority road, stop, children crossing and go straight or right are shown below:

Speed limit (50 km/h)

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .92993         			| No passing for vehicles over 3.5 metric tons   									|
| .06766     				| Ahead only 										|
| .00165					| Vehicles over 3.5 metric tons prohibited											|
| .00064	      			| Speed limit (80 km/h)					 				|
| .00004				    | Turn right ahead      							|


Priority Road

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000   			| Priority road   									|
| .00000     				| Roundabout mandatory 										|
| .00000					| Yield											|
| .00000	      			| Keep right					 				|
| .00000				    | Speed limit (100 km/h)      							|


Stop

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000         			| Stop   									|
| .00000     				| Speed limit (100 km/h) 										|
| .00000					| Keep right											|
| .00000	      			| Turn left ahead					 				|
| .00000				    | Turn right ahead      							|


Children crossing

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .94658         			| Road narrows on the right   									|
| .03416     				| Children crossing 										|
| .01473					| Beware of ice/snow											|
| .00285	      			| Dangerous curve to the right					 				|
| .00251				    | Road work      							|


Go straight or right

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00000         			| Go straight or right   									|
| .00000     				| Ahead only										|
| .00000					| Speed limit (60 km/h)											|
| .00000	      			| End of all speed and passing limits					 				|
| .00000				    | Vehicles over 3.5 metric tons prohibited      							|

The signs correctly detected obtain a 100% of accuracy. For the children crossing one, it is detected as children crossing with a probability of 3.4% (the second most detected one). However, for the signal limit of 50%, curiously it is not even within the 5 highest probabilities.

### Neural network visualization

The visualization of the first convolution for the No passing sign is shown below:

![alt text][image14]

Regarding the second one:

![alt text][image15]

The first convolution shows that the boundaries of the sign are firstly detected. Regarding the second one, the low amount of pixels (5x5) makes that it is difficult for the human eye to discern which features are detected. However it is expected to detect the details of the image (in the case of this traffic sign, both cars)

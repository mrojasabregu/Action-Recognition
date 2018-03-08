# Overview
This is the initial prototype of a computationally viable action recognition algorithm. Initial tests for an action recognition algorithm on Weizmann data-set is presented. 
It results in a significant 80 – 90 % and good frame rate (16 fps) accuracy on a mediocre CPU despite the present methods using high-end GPUs.
It uses optical flow and two-dimensional Convolutional Neural Network for extracting spatial features from a video, pipelined with one-dimensional Convolutional Neural Network extracting temporal features and finally fed to a dense neural network for classification. 
The plan is to develop a computationally viable product to act on a real-time basis. 
# Procedure
Unlike the existing deep learning action recognition backpropagating and claculating the gradients from end to end, the algorithm uses two separate
networks that can be trained piecewise. This reduces the computational complexity and improves the training accuracy/speed at the same time.
Keeping in mind the computational complexity of the task, the algorithm is as follows:
## Spatial Feature Extraction:
   The first step is to extract the frames from the given video and assign label to the action for training the neural network. 
   The videos typically contain features in two dimensions, spatial and temporal. This step extracts the spatial information from the frames
   via a two dimensional neural network [2D-CNN](https://github.com/RameenAbdal/Action-Recognition/blob/master/twoD_Train.py).
   Instead of using the pretrained models (e.g., PASCAL VOC), the model is trained from the scratch using custom designed model to counter 
   the computational complexity and making the process easily manageable. Before feeding it to a 2D-CNN model the necessary preprocessing is
   performed. During the expirementations, following problems were faced during the training:
   
   1. Considering the video data to be 3D (**_f_(x,y,t)**), the extracted frames are typically 2D (**_f_(x,y)**) with RGB channels. 
         While training the 2D-CNN on such data, it was observed that the model tends to overfit on the background (redundant feature for
         an action recognition model). To avoid this, a background subtraction algorithm is used to segment the action from the background ([see myGenerator](https://github.com/RameenAbdal/Action-Recognition/blob/master/Functions_2.py)).
         Also, the segmented spatial data tend to learn the features of the action performer rather than the action itself. To make the model more
         generic, the transformation was performed on the frames using the Optical Flow. The vectors from the Optical Flow algorithm are converted
         into a color coded scheme, assigning color to each direction of vectors and the associated magnitude as color intensity ([see myGenerator_optical](https://github.com/RameenAbdal/Action-Recognition/blob/master/Functions_2.py)). After this
         operation, the data typically represent color coded frames representing an action irrespective of the features of a 
         action performer. This improves the training accuracy and avoids overfitting.
   
    2. The spatial size of the videos is an important factor determining the training and predicting speeds and accuracy. The aim is to resize
         the videos to lower dimensions ensuring minimum information loss. It is observed that the color coded sequence provides good accuracy 
         on 2D-CNN model even if the spatial size is reduced by considerable amount. This hepled in reducing the computational complexity of
         the algorithm.
 ## Temporal Feature Extraction:
   After saving the weights of the 2D-CNN model, the color coded frames are fed to the model one by one and the feature maps from the last
    convolutional layer are extracted. These 2D feature maps are converted into 1D vectors (flattening and mapping the pixel positions from 2D to a 1D vector).
    For _N_ such frames _N_ number of such vectors are extracted (e.g., for a 50 frame video, 50 such vectors are extracted and compiled over
    one another forming a sequence). In this way the 3D video data is represented as a 2D sequence of vectors. Specifying a static sequence size,
    a one dimensional neural network ([1D-CN](https://github.com/RameenAbdal/Action-Recognition/blob/master/Train_1d.py)) is used to extract temporal features from these sequences. In this way two separate pipelined networks used.
    One acts as a decoder extracting mainly the spatial information and the other, temporal information. 
# Results
The [test.py](https://github.com/RameenAbdal/Action-Recognition/blob/master/test.py) tests the input video for the actions. It also specifies the frame rate, accuracy and the predicted classes. Watch the performace:

[![Alt text for your video](https://github.com/RameenAbdal/Action-Recognition/blob/master/Results_appendingframes/1.png) <!-- .element height="10%" width="10%" -->](https://github.com/RameenAbdal/Action-Recognition/blob/master/Results_appendingframes/bend_frame.avi?raw=true)
[![Alt text for your vide](https://github.com/RameenAbdal/Action-Recognition/blob/master/Results_appendingframes/2.png) <!-- .element height="10%" width="10%" -->](https://github.com/RameenAbdal/Action-Recognition/blob/master/Results_appendingframes/side.avi?raw=true)
[![Alt text for your vid](https://github.com/RameenAbdal/Action-Recognition/blob/master/Results_appendingframes/3.png) <!-- .element height="10%" width="10%" -->](https://github.com/RameenAbdal/Action-Recognition/blob/master/Results_appendingframes/wave1.avi?raw=true)

The work in progess for fight detection:

   ![Alt text for your vi](https://github.com/RameenAbdal/Action-Recognition/blob/master/Results_appendingframes/4.png) 
   ![Alt text for your i](https://github.com/RameenAbdal/Action-Recognition/blob/master/Results_appendingframes/5.png) 

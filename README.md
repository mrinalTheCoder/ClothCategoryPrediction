# Cloth Category Prediction

## Introduction
This repo is the code for DeepFashion category prediction. 
Given a picture of an article of clothing, we must classify it into one of 45 types. 
The dataset used to train the model is the [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) dataset.
I have used keras for this task, and the model was trained on Google Colab.

## Methods Used
### Transfer leaarning
This method has been tried in [classify_clothes_xception.ipynb](https://github.com/mrinalTheCoder/ClothCategoryPrediction/blob/master/classify_clothes_xception.ipynb)
In it, we get the architecture of the xcpetion image classification model trained on the imagennet dataset. 
To the last convolutional dataset, we add our own Dense and Dropout layers.

### Attention network
![alt text](https://github.com/fdjingyuan/Deep-Fashion-Analysis-ECCV2018/raw/master/images/network.png) <br>
Above is a diagramatic representation of the network used for this task. 
In it, the last convolutional layer of the VGG16 model is split into 3 branches: the landmark branch, attention branch and VGG networks.
Several convolutional layers and trannspose convolution layers are applied to upscale and downscale the data.
In the end, it is flattened and fed into a Dense layer, with 1000 neurons using relu activation. 
The final layer has 45 neurons (the number of categories) and uses softmax activation.
Categorical crossentropy loss and Adam optimizer with an exponential decay learning rate schedule were used. 
This was first used in a paper [here](https://drive.google.com/file/d/1Dyj0JIziIrTRWMWDfPOapksnJM5iPzEi/view).<br

## Training the model
The model was trained in Google Colab, using a GPU. To speed up training, the processing of input data has been parallelised.
TensorFlow's dataset API is used for this task. 
Since we cannot store all the data at once in RAM, we first make a `tf.data.Dataset` object containing the filepaths and labels.
We then call a function which reads the image at the filepath, applies the necessary augmentations and returns it for training.

## Results
|    **Method**   | **Crossentropy loss** | **top 3 accuracy** | **top accuracy** |
|:---------------:|:---------------------:|:------------------:|:----------------:|
|Transfer learning|1.38                   |83%                 |61%               |
|Attention network|1.37                   |83.25%              |61%               |

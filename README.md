# AI-hackathon-
# About the Project:
Detecting and recognizing text in natural images is a challenging problem that has recently received attention. Most methods attempting this have relied on elaborate models incorporating carefully hand crafted features or large amounts of prior knowledge. All methods assume the knowledge of a lexicon of 20-30 words that form the super set of all possible words that can occur in a test image. This makes the problem a lot easier than it would other- wise be. In this report, we present a method of detecting as well as recognizing text contained in natural images using Convolutional Neural Networks (CNNs). We use two CNNs, one trained for the character detection task and another for the character recognition task, as building blocks for the algorithm. Significantly, we do not assume any prior knowledge of a lexicon, unlike all previous works. We demonstrate state-of-the-art experimental results for three different tasks: character detection, character recognition and word recognition, while also showing qualitative results for text recognition. 

# Built With
1.	Keras
2.	Tensorflow
3.	json

# Our Approach 
Deviating from some of the above approaches that rely on hand-crafted features, we use CNNs for this problem. We now briefly give an overview of our system. We tackle the problems of detection and recognition in two cascaded stages. Given a novel image instance, we consider a sliding window over the whole span of the image at multiple scales . For each of these windows, we con- sider the image patch within that window and pass it to the character detection engine. We train the detection CNN in such a way that this character detection engine outputs a high score if there is a centred character in the input im- age patch. For these firings from the detection engine, we pass the corresponding patches to the recognition engine to determine which character is present in them. We use two Convolutional Neural Networks, one for the detection part and another for the recognition part. Also, we do not make the assumption that a lexicon which contains all words that could occur in a given test image exists. 

 # Detection of CNN 
The detection CNN takes as input a 64x64 image patch and outputs the probabilities associated with two classes: character present and character absent. We call the probability associated with the character present class as the ‘detection score’ of that input patch. 

# Getting Started
Step 1. Clone the repository.
Step 2. Download the dataset from https://drive.google.com/drive/folders/1O8TT0s4zMyiI6zR-biVRoiLiAUy-W1H0 and place it in the respective data file. Remember both the translation pipelines have different data folder

# Installation
1. Python 3.7
2. Install python libraries
conda install -r requirements.txt
, import os
, import numpy as np
, import pandas as pd
, from PIL import Image
, from matplotlib import pyplot as plt
3. import json
4. import tensorflow as tf
5. from sklearn.utils import shuffle
6. from sklearn.preprocessing import LabelEncoder
7. from sklearn.model_selection import train_test_split
8. from tensorflow.keras.models import Model
9. from tensorflow.keras import layers
10. from tensorflow.keras import optimizers, losses, metrics
11. from tensorflow.keras.preprocessing.image import ImageDataGenerator
12. from tensorflow.keras.applications import vgg16
13. from tensorflow.keras.applications.vgg16 import preprocess_input
14. from tensorflow.keras.losses import BinaryCrossentropy
15. from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
16. from tensorflow.keras.preprocessing import image

# Training
As explained before, we use two CNNs, one for the char- acter detection task and another for the character recognition task. We have used 2 CNN architectures in total, and we label them as CNN-1 and CNN-2. We use rectified linear units (ReLU) as our non-linearities and use dropout as regular- ization. 

# References
1.	Attention is all you need paper»https://arxiv.org/abs/1706.03762
2.	http://cs231n.stanford.edu/reports/2015/pdfs/vikesh_final.pdf

Made By

•	Contact Abhinav Chandra @abhinavchandra0526@gmail.com




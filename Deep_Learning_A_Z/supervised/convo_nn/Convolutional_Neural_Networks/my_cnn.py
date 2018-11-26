import pandas as pd

# Part 1
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

## Initialize the CNN
classifier = Sequential()

## Step 1 - Convolution Layer
## Use 32 filters of 3 rows and 3 columns each
## Use default border mode
## Specify that all images are 3 channel (one for each primary color) images of 64 x 64 pixels
## Use rectifier as activation function
classifier.add(Convolution2D(32, 3, 3, 
                             border_mode = 'same', 
                             input_shape = (64, 64, 3), 
                             activation = 'relu' ))

## Step 2 - Max Pooling Layer



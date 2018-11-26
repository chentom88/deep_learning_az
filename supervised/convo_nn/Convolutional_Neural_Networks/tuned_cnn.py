# Part 1 - Build the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

## Initialize the CNN
classifier = Sequential()

## Step 1 - Convolution Layer
classifier.add(Convolution2D(32, 3, 3, 
                             border_mode = 'same', 
                             input_shape = (64, 64, 3), 
                             activation = 'relu' ))

## Step 2 - Max Pooling Layer
## Specify pool size of 2 x 2 for max summation
classifier.add(MaxPooling2D( pool_size = (2, 2) ))

## Can improve performance by adding another convolutional layer
## Since input is from pooled samples, don't need to specify input shape
## as Keras will have the shape
classifier.add(Convolution2D(32, 3, 3, 
                             border_mode = 'same', 
                             activation = 'relu' ))
classifier.add(MaxPooling2D( pool_size = (2, 2) ))

## Step 3 - Flattening
classifier.add(Flatten())

## Step 4 - Full Connection

### Add hidden layer
### Number of hidden nodes (128) was arbitrarily selected
### Use rectifier as activation again
classifier.add(Dense(output_dim = 128, 
                    activation = 'relu'))

## Can also improve performance by adding another hidden layer

### Add output layer
### Use sigmoid function as activation
classifier.add(Dense(output_dim = 1,
                     activation = 'sigmoid'))

## Compile the CNN
## Use the adam stochastic descent algorithm
## Use the binary cross entropy function for the loss function because this is
## a logistic regression classifying a binary output
## Use accuracy for metrics function
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# Part 2 - Fit the CNN to the images

## Need this for MacOS error about libiomp5.dylib
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


## Import ImageDataGenerator that will perform 
## image augmentation (random transformations to increase
## data sample size from current set of images)
from keras.preprocessing.image import ImageDataGenerator

## Creating data augmenter for training images
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

## Create data augmenter for test images
test_datagen = ImageDataGenerator(rescale = 1./255)

## Point training augmenter to training set
## class mode is 'binary' because it's a binary classification
training_set = train_datagen.flow_from_directory('dataset/training_set', 
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')
## Point training augmenter to test set
## class mode is 'binary' because it's a binary classification
test_set = test_datagen.flow_from_directory('dataset/test_set', 
                                                  target_size = (64, 64),
                                                  batch_size = 32,
                                                  class_mode = 'binary')

## Fit the classifier to the augmented images
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)
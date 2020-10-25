#Careful when running this on Mac

import os
import zipfile
import random
import tensorflow as tf
import shutil
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

print(len(os.listdir('tmp/PetImages/Cat/')))
print(len(os.listdir('tmp/PetImages/Dog/')))

try:
    #YOUR CODE GOES HERE
    parent_dir = 'tmp'
    directory = 'cats-v-dogs'
    base_dir = os.path.join(parent_dir, directory)
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'training')
    validation_dir = os.path.join(base_dir, 'testing')
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    # Directory with our training cat/dog pictures
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)

    # Directory with our validation cat/dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_cats_dir)
    os.mkdir(validation_dogs_dir)
except OSError:
    pass


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    source_list = os.listdir(SOURCE)
    train_data = []
    test_data = []
    random.sample(source_list,len(source_list))

    for element in os.listdir(SOURCE):
        if (len(train_data) < len(source_list)*SPLIT_SIZE):
            train_data.append(element)
        else:
            test_data.append(element)
    
    for element in train_data:
        source_train = SOURCE + element
        training = TRAINING + element
        copyfile(source_train, training)

    for element in test_data:
        source_test = SOURCE + element
        testing = TESTING + element
        copyfile(source_test, testing)



CAT_SOURCE_DIR = "tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir('tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('tmp/cats-v-dogs/testing/dogs/')))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer=RMSprop(lr=0.001),
            loss='binary_crossentropy',
            metrics=['acc'])

TRAINING_DIR = "tmp/cats-v-dogs/training"
train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "tmp/cats-v-dogs/testing"
validation_datagen = ImageDataGenerator(
                                        rescale=1./255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                            batch_size=10,
                                                            class_mode='binary',
                                                            target_size=(150, 150))

history = model.fit_generator(train_generator,
                              epochs=20,
                              verbose=1,
                              validation_data=validation_generator)

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)
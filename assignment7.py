import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

def get_data(filename):

    with open(filename) as training_file:
        images = []
        labels =[]
        training_file.readline()
        for line in training_file:
            line = line.split(',')
            labels.append(line[0])
            images.append(np.array_split(line[1:785], 28))
        labels = np.asarray(labels).astype(int)
        images = np.asarray(images).astype(int)
    return images, labels

path_sign_mnist_train = f"{getcwd()}/tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

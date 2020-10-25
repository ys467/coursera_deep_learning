import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd


with open("tmp2/test.csv") as training_file:
        images = []
        labels =[]
        tmp = []
        data = np.genfromtxt(training_file, delimiter=',')
        print(training_file.readline())
        print(training_file)
        dim = data.shape
        for i in range (0, data.shape[0]):
            if i == 0:
                labels.append(data[i])
            else:
                images.append(np.array_split(data[i],3))
        print(type(labels))
        labels = np.asarray(labels).astype(int)
        images = np.asarray(images).astype(int)
        print(images)
        print(images.size)

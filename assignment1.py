#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:42:33 2020

@author: yungyu
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model(y_new):
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    ys = np.array([100000, 150000, 200000, 250000, 300000], dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1,
                                                    input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    return model.predict(y_new)[0]

prediction = house_model([7.0])
print(prediction)

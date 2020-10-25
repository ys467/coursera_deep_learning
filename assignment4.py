import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

#path = f"{getcwd()}/tmp2/happy-or-sad.zip"

#zip_ref = zipfile.ZipFile(path, 'r')
#zip_ref.extractall("/tmp/h-or-s")
#zip_ref.close()

def train_happy_sad_model():
    
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>DESIRED_ACCURACY):
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True

    callback = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    from tensorflow.keras.optimizers import RMSprop

    model.compile(optimizer=RMSprop(lr=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(
        'tmp2/happy-or-sad/',
        target_size=(150,150),
        batch_size=128,
        class_mode='binary')
    
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,  
        epochs=15,
        callbacks = [callback],
        verbose=1)
    
    return history.history['accuracy'][-1]

train_happy_sad_model()    
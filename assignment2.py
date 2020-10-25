import tensorflow as tf
from os import path, getcwd, chdir

path = f"{getcwd()}/mnist.npz"

def train_mnist():

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    mnist = tf.keras.datasets.mnist

    callback = myCallback()

    (x_train, y_train), (x_test, y_test)=mnist.load_data(path=path)
    x_train, x_test = x_train/255.0, x_test/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10, callbacks=[callback])

    return history.epoch , history.history['accuracy'][-1]

train_mnist()
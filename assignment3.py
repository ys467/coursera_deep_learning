import tensorflow as tf
from os import path, getcwd, chdir

path = f"{getcwd()}/mnist.npz"

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

def train_mnist_conv():

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True

    mnist = tf.keras.datasets.mnist

    callback = myCallback()

    (x_train, y_train), (x_test, y_test)=mnist.load_data(path=path)
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_train = x_train / 255.0
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_test = x_test/255.0

    model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()

    history=model.fit(
        x_train, y_train, epochs=20, callbacks=[callback]
    )

    evaluation = model.evaluate(x_test, y_test)

    return history.epoch , history.history['accuracy'][-1], evaluation

train_mnist_conv()
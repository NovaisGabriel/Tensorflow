import tensorflow as tf
import numpy as np

from tensorflow import keras


# class MyCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if logs.get('accuracy') >= 0.6:  # Experiment with changing this value
#             print("\nReached 60% accuracy so cancelling training!")
#             self.model.stop_training = True


# callbacks = MyCallback()

callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.1,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False
)

fashion_mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255

model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, validation_data=(test_images, test_labels), epochs=10, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

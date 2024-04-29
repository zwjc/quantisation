import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import time

# Function to load and preprocess the MNIST dataset
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'constant')
    x_train = x_train.reshape((-1, 32, 32, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 32, 32, 1)).astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

# Custom callback to log epoch times
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_start_time)

# Function to build the LeNet model
def build_lenet_model():
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=(32, 32, 1), padding='valid'),
        layers.AveragePooling2D(),
        layers.Conv2D(16, kernel_size=(5, 5), activation='tanh', padding='valid'),
        layers.AveragePooling2D(),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main function to run the workflow
def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = build_lenet_model()
    
    time_callback = TimeHistory()
    model.fit(x_train, y_train, epochs=2, validation_split=0.1, batch_size=32, verbose=2, callbacks=[time_callback])

    # Print the metrics and runtimes
    for i, time in enumerate(time_callback.times):
        print(f'Epoch {i+1} - Runtime: {time:.2f} seconds')

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f'Original Model - Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()


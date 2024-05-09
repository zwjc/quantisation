import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time
import os

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

def evaluate_original_model(model, x_test, y_test):
    start_time = time.time()
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    total_inference_time = time.time() - start_time
    average_inference_time = total_inference_time / len(x_test)
    return accuracy, average_inference_time

def convert_to_tflite(model, quantize=True):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model

def evaluate_model(tflite_model, x_test, y_test):
    # Save the TFLite model to file
    tflite_model_path = 'lenet_quantized.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference on TFLite model and calculate accuracy
    prediction_digits = []
    start_time = time.time()
    for test_image in x_test:
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        digit = np.argmax(output_data)
        prediction_digits.append(digit)
    total_inference_time = time.time() - start_time
    average_inference_time = total_inference_time / len(x_test)

    # Calculate accuracy
    accurate_count = sum([int(y_pred == np.argmax(y_true)) for y_pred, y_true in zip(prediction_digits, y_test)])
    accuracy = accurate_count / len(y_test)

    return accuracy, average_inference_time

def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = build_lenet_model()
    model.fit(x_train, y_train, epochs=10, validation_split=0.1, verbose=2)
    original_accuracy, original_inference_time = evaluate_original_model(model, x_test, y_test)
    
    tflite_model = convert_to_tflite(model, quantize=True)
    tflite_accuracy, tflite_inference_time = evaluate_model(tflite_model, x_test, y_test)
    
    original_model_size = os.path.getsize('original_lenet_model.h5')  # Provide path if model is saved
    tflite_model_size = os.path.getsize('lenet_quantized.tflite')
    
    print(f"Original Model Size: {original_model_size / 1024:.2f} KB")
    print(f"TFLite Model Size: {tflite_model_size / 1024:.2f} KB")
    print(f"Original Model Accuracy: {original_accuracy * 100:.2f}%")
    print(f"TFLite Model Accuracy: {tflite_accuracy * 100:.2f}%")
    print(f"Original Average Inference Time per Sample: {original_inference_time * 1000:.2f} ms")
    print(f"TFLite Average Inference Time per Sample: {tflite_inference_time * 1000:.2f} ms")

if __name__ == '__main__':
    main()

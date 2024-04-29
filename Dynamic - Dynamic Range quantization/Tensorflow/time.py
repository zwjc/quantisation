import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import time
import os

# Function to load and preprocess the MNIST dataset
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant', constant_values=0)
    x_test = x_test.reshape((-1, 32, 32, 1)).astype('float32') / 255.0
    y_test = to_categorical(y_test, 10)
    return x_test, y_test

# Load TensorFlow and TensorFlow Lite models
def load_models():
    # Load the original model
    original_model = tf.keras.models.load_model('original_lenet_model.h5')
    
    # Load the quantized TFLite model
    interpreter = tf.lite.Interpreter(model_path='lenet_quantized.tflite')
    interpreter.allocate_tensors()
    
    return original_model, interpreter

# Function to run inference and measure runtime
def measure_inference_time(model, data, is_tflite=False):
    start_time = time.time()
    
    if is_tflite:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Assuming single batch inference for simplicity
        for test_image in data:
            model.set_tensor(input_details[0]['index'], np.expand_dims(test_image, axis=0))
            model.invoke()
            output_data = model.get_tensor(output_details[0]['index'])
    else:
        predictions = model.predict(data)
    
    end_time = time.time()
    return end_time - start_time

# Main execution function
def main():
    x_test, y_test = load_and_preprocess_data()
    original_model, tflite_interpreter = load_models()
    
    # Measure inference time for the original model
    original_time = measure_inference_time(original_model, x_test)
    print(f'Original model inference time: {original_time:.4f} seconds')
    
    # Measure inference time for the quantized model
    quantized_time = measure_inference_time(tflite_interpreter, x_test, is_tflite=True)
    print(f'Quantized model inference time: {quantized_time:.4f} seconds')

    # Calculate and print the difference
    difference = original_time - quantized_time
    print(f'Inference time difference: {difference:.4f} seconds')

if __name__ == '__main__':
    main()


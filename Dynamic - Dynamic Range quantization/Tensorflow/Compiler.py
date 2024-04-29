import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Function to load and preprocess data
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant', constant_values=0)
    x_test = x_test.reshape((-1, 32, 32, 1)).astype('float32') / 255.0
    y_test = to_categorical(y_test, 10)
    return x_test, y_test

# Load the TFLite model and allocate tensors
def load_tflite_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

# Run inference using the TFLite model
def run_tflite_inference(interpreter, x_test):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for test_image in x_test:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest probability.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        digit = np.argmax(output_data)
        prediction_digits.append(digit)

    return prediction_digits

# Calculate model accuracy
def calculate_accuracy(y_true, y_pred):
    # Compare prediction results with ground truth labels to calculate accuracy.
    accurate_count = 0
    for index in range(len(y_pred)):
        if y_pred[index] == np.argmax(y_true[index]):
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(y_pred)
    return accuracy

# Main execution
def main():
    x_test, y_test = load_and_preprocess_data()
    interpreter = load_tflite_model('lenet_quantized.tflite')
    predictions = run_tflite_inference(interpreter, x_test)
    accuracy = calculate_accuracy(y_test, predictions)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()


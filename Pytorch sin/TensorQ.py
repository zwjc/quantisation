import numpy as np
import tensorflow as tf
import time

NUM_POINTS = 1000  # Number of points between 0 and 2*pi

def compute_sine_tensorflow(angles, dtype=tf.float32):
    start = time.time()
    angles_tf = tf.convert_to_tensor(angles, dtype=dtype)
    sine_tf = tf.math.sin(angles_tf)
    end = time.time()
    print(f"TensorFlow computation time with dtype {dtype}: {end - start:.6f} seconds")
    return sine_tf.numpy()  # Convert TensorFlow tensor to NumPy array for error calculation and easy handling

def calculate_error(sine_np, sine_tf):
    absolute_error = np.abs(sine_np - sine_tf)
    mean_absolute_error = np.mean(absolute_error)
    return mean_absolute_error

# Create an array of angles from 0 to 2*pi
angles = np.linspace(0, 2 * np.pi, NUM_POINTS, dtype=np.float32)

# Compute sine using TensorFlow with default float32 precision
sine_tf_fp32 = compute_sine_tensorflow(angles, dtype=tf.float32)

# Compute sine using TensorFlow with reduced precision (float16)
sine_tf_fp16 = compute_sine_tensorflow(angles, dtype=tf.float16)

# Compute sine using NumPy for comparison
sine_np = np.sin(angles)

# Calculate errors
error_fp32 = calculate_error(sine_np, sine_tf_fp32)
error_fp16 = calculate_error(sine_np, sine_tf_fp16)

print(f"Average absolute error between NumPy and TensorFlow float32: {error_fp32}")
print(f"Average absolute error between NumPy and TensorFlow float16: {error_fp16}")

# Memory usage estimation
print("Estimated memory usage for TensorFlow tensor (float32):", sine_tf_fp32.nbytes, "bytes")
print("Estimated memory usage for TensorFlow tensor (float16):", sine_tf_fp16.nbytes, "bytes")


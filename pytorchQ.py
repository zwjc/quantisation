import numpy as np
import torch
import time
import sys

NUM_POINTS = 1000  # Number of points between 0 and 2*pi

def compute_sine_numpy(angles):
    start = time.time()
    sine_np = np.sin(angles)
    end = time.time()
    print(f"NumPy computation time: {end - start:.6f} seconds")
    return sine_np

def compute_sine_torch(angles, dtype=torch.float32):
    start = time.time()
    angles_torch = torch.tensor(angles, dtype=dtype)
    sine_torch = torch.sin(angles_torch)
    end = time.time()
    print(f"PyTorch computation time with dtype {dtype}: {end - start:.6f} seconds")
    return sine_torch.numpy()

def calculate_error(sine_np, sine_torch):
    absolute_error = np.abs(sine_np - sine_torch)
    mean_absolute_error = np.mean(absolute_error)
    return mean_absolute_error

# Create an array of angles from 0 to 2*pi
angles = np.linspace(0, 2 * np.pi, NUM_POINTS, dtype=np.float32)


sine_np = compute_sine_numpy(angles)


sine_torch_fp32 = compute_sine_torch(angles)


sine_torch_fp16 = compute_sine_torch(angles, dtype=torch.float16)


error_fp32 = calculate_error(sine_np, sine_torch_fp32)
error_fp16 = calculate_error(sine_np, sine_torch_fp16)

print(f"Average absolute error between NumPy and PyTorch float32: {error_fp32}")
print(f"Average absolute error between NumPy and PyTorch float16: {error_fp16}")

# Estimate memory usage
print("Memory usage for NumPy array (float32):", sine_np.nbytes, "bytes")
print("Memory usage for PyTorch tensor (float32):", sine_torch_fp32.nbytes, "bytes")
print("Memory usage for PyTorch tensor (float16):", sine_torch_fp16.nbytes, "bytes")


import torch
import numpy as np
import time
import os

NUM_POINTS = 1000  # Number of points between 0 and 2*pi

class SineModel(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)

def compute_sine_pytorch(angles, dtype=torch.float32, quantize=False):
    model = SineModel()
    if quantize:
        # Note: Quantization is not effective here as there are no weights to quantize
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model = torch.quantization.prepare(model, inplace=False)
        model = torch.quantization.convert(model, inplace=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    angles_torch = torch.tensor(angles, device=device, dtype=dtype)

    start = time.time()
    with torch.no_grad():
        sine_torch = model(angles_torch)
    end = time.time()

    print(f"{'Quantized' if quantize else 'Non-quantized'} PyTorch computation on {device} time with dtype {dtype}: {end - start:.6f} seconds")
    return sine_torch.cpu().numpy()

def calculate_error(sine_np, sine_torch):
    absolute_error = np.abs(sine_np - sine_torch)
    mean_absolute_error = np.mean(absolute_error)
    return mean_absolute_error

# Create an array of angles from 0 to 2*pi
angles = np.linspace(0, 2 * np.pi, NUM_POINTS, dtype=np.float32)

# Compute sine using NumPy for baseline comparison
sine_np = np.sin(angles)

# Compute sine using PyTorch non-quantized
sine_torch = compute_sine_pytorch(angles)

# Compute sine using PyTorch quantized (Note: this won't actually quantize the sine operation)
sine_torch_quantized = compute_sine_pytorch(angles, quantize=True)

# Calculate errors
error_torch = calculate_error(sine_np, sine_torch)
error_torch_quantized = calculate_error(sine_np, sine_torch_quantized)

print(f"Average error for non-quantized PyTorch: {error_torch}")
print(f"Average error for quantized PyTorch: {error_torch_quantized}")

# Memory usage estimation and comparison
def print_model_size(model, description="Model"):
    torch.save(model.state_dict(), 'temp.p')
    size = os.path.getsize('temp.p')
    print(f"{description} size: {size} bytes")
    os.remove('temp.p')

model_non_quantized = SineModel()
print_model_size(model_non_quantized, "Non-quantized model")

model_quantized = torch.quantization.quantize_dynamic(SineModel(), {torch.nn.Module}, dtype=torch.qint8)
print_model_size(model_quantized, "Quantized model")



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.quantization
import time
import os

# Define the LeNet architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(400, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )
   
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x

# Load and transform data
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjust if needed to match TensorFlow's [0,1] scaling
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

# Instantiate the model and optimizer
model = LeNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

print("Training original model:")
train_model(model, train_loader, criterion, optimizer)

# Apply Dynamic Quantization
dynamic_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

def evaluate_and_print(model, model_name, data_loader):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for data, targets in data_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    end_time = time.time()
    accuracy = 100 * correct / total
    inference_time = end_time - start_time  # Corrected variable name
    print(f'{model_name} - Accuracy: {accuracy:.2f}%, Inference Time: {inference_time:.3f} seconds')

# Save and evaluate models
torch.save(model.state_dict(), 'original_model.pth')
torch.save(dynamic_model.state_dict(), 'dynamic_quantized_model.pth')

# Calculate file sizes
original_size = os.path.getsize('original_model.pth')
quantized_size = os.path.getsize('dynamic_quantized_model.pth')

# Evaluate both models
print("Evaluating original model:")
evaluate_and_print(model, "Original Model", test_loader)
print("Evaluating dynamic quantized model:")
evaluate_and_print(dynamic_model, "Dynamic Quantized Model", test_loader)

# Print model sizes
print(f'Original Model Size: {original_size} bytes')
print(f'Quantized Model Size: {quantized_size} bytes')
print(f'Size Reduction: {100 * (1 - quantized_size / original_size):.2f}%')

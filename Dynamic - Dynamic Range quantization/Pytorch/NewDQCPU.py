import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time
import torch.quantization

# Define LeNet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.log_softmax(x, dim=1)

# Function to load and preprocess the MNIST dataset
def load_and_preprocess_data():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader

# Function to train the model
def train_model(model, train_loader, epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        end_time = time.time()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Time: {end_time - start_time} seconds")

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    inference_time = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            inference_time += end_time - start_time
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy, inference_time

def main():
    train_loader, test_loader = load_and_preprocess_data()
    model = LeNet()
    train_model(model, train_loader)

    torch.save(model.state_dict(), 'original_lenet_model.pth')
    original_model_size = os.path.getsize('original_lenet_model.pth')

    # Quantize the model
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), 'quantized_lenet_model.pth')
    quantized_model_size = os.path.getsize('quantized_lenet_model.pth')

    accuracy, inference_time = evaluate_model(model, test_loader)
    print(f"Original Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Original Total Inference Time: {inference_time:.2f} seconds")
    print(f"Original Average Inference Time per Sample: {inference_time / len(test_loader.dataset) * 1000:.2f} ms")
    print(f"Original Model Size: {original_model_size / 1024:.2f} KB")

    quantized_accuracy, quantized_inference_time = evaluate_model(quantized_model, test_loader)
    print(f"Quantized Model Accuracy: {quantized_accuracy * 100:.2f}%")
    print(f"TFLite Total Inference Time: {quantized_inference_time:.2f} seconds")
    print(f"TFLite Average Inference Time per Sample: {quantized_inference_time / len(test_loader.dataset) * 1000:.2f} ms")
    print(f"Quantized Model Size: {quantized_model_size / 1024:.2f} KB")

if __name__ == '__main__':
    main()

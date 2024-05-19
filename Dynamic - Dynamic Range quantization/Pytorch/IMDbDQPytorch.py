import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

# Set seed for reproducibility
set_seed(0)

# Helper functions
def get_average_inference_time(model, data, device):
    model.to(device)
    with torch.no_grad():
        # Measure inference time
        start = time.time()
        for i in range(5):  # Number of measurement iterations
            model(input_ids=data[0].to(device), attention_mask=data[1].to(device))
            print(f"Inference iteration {i+1}/5 completed.")
        end = time.time()
        average_inference_time = (end - start) / 5 * 1000
    
    return average_inference_time

def plot_speedup(inference_time_stock, inference_time_optimized):
    data = {'FP32': inference_time_stock, 'INT8': inference_time_optimized}
    model_type = list(data.keys())
    times = list(data.values())
    fig = plt.figure(figsize=(10, 5))
    plt.bar(model_type, times, color='blue', width=0.4)
    plt.ylabel("Runtime (ms)")
    plt.title(f"Speedup achieved - {inference_time_stock / inference_time_optimized:.2f}x")
    plt.savefig('speedup.png')
    plt.show()

# Load pretrained model and tokenizer
model_name = "JiaqiLee/imdb-finetuned-bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load dataset
data = load_dataset("imdb")

text = data['test']['text']
labels = data['test']['label']

# Define dataset class
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, data_size):
        self.texts = texts[:data_size]
        self.labels = labels[:data_size]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return inputs, label

# Define evaluation function
def eval_func(model_q, test_loader):
    model_q.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for _, batch in enumerate(test_loader):
            inputs, labels = batch
            ids = inputs['input_ids'].squeeze(1)
            mask = inputs['attention_mask'].squeeze(1)
            outputs = model_q(input_ids=ids, attention_mask=mask)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.numpy())
    return accuracy_score(test_labels, test_preds)

# Choose device (CPU)
device = torch.device('cpu')

# Define the desired size of the subset
subset_size = 500  # Choose the number of samples you want in the subset

# Sample a subset of the data
text_subset = text[:subset_size]
labels_subset = labels[:subset_size]

# Create the IMDBDataset with the subset
test_dataset = IMDBDataset(text_subset, labels_subset, tokenizer=tokenizer, data_size=subset_size)

# Create dataloader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

# Get a batch of data
batch = next(iter(test_loader))
data = (batch[0]['input_ids'].squeeze(1), batch[0]['attention_mask'].squeeze(1))

# Benchmark stock PyTorch model
model.eval()
print("Benchmarking FP32 model...")
inference_time_stock = get_average_inference_time(model, data, device)
print(f"Time taken for forward pass (FP32): {inference_time_stock} ms")

# Calculate accuracy of FP32 model
print("Calculating accuracy of FP32 model...")
accuracy_fp32 = eval_func(model, test_loader)
print(f"Accuracy of FP32 model: {accuracy_fp32}")

# Get FP32 model size
fp32_model_size = sum(p.numel() for p in model.parameters())
print(f"Size of FP32 model (parameters): {fp32_model_size}")

# Perform dynamic quantization
print("Performing dynamic quantization...")
model_int8 = torch.quantization.quantize_dynamic(
    model,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8  # the target dtype for quantized weights
)

# Measure the inference time for the quantized model
print("Benchmarking INT8 model...")
inference_time_optimized = get_average_inference_time(model_int8, data, device)
print(f"Time taken for forward pass (INT8): {inference_time_optimized} ms")

# Calculate accuracy of INT8 model
print("Calculating accuracy of INT8 model...")
accuracy_int8 = eval_func(model_int8, test_loader)
print(f"Accuracy of INT8 model: {accuracy_int8}")

# Get INT8 model size
int8_model_size = sum(p.numel() for p in model_int8.parameters())
print(f"Size of INT8 model (parameters): {int8_model_size}")

# Plot performance gain
plot_speedup(inference_time_stock, inference_time_optimized)

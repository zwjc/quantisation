import torch
from torch.utils.data import DataLoader
from time import time
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['sentence'], padding="max_length", truncation=True, max_length=128)

def load_and_prepare_data(tokenizer):
    dataset = load_dataset('glue', 'sst2')
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_datasets['validation'].shuffle(seed=42).select(range(300))
    return train_dataset, eval_dataset

def train_model(model, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    return model

def save_model(model, path, quantized=False):
    if quantized:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(model.state_dict(), path)

def load_model(model_path, quantized=False):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    state_dict = torch.load(model_path)
    if quantized:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def evaluate_model(model, dataloader):
    total, correct = 0, 0
    start_time = time()
    for batch in dataloader:
        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        labels = batch['label']
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    elapsed_time = time() - start_time
    accuracy = correct / total
    return accuracy, elapsed_time

def compare_models(dataset, batch_size=32):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Load models
    original_model = load_model('temp.p')
    quantized_model = load_model('temp_quantized.p', quantized=True)

    # Evaluate models
    original_accuracy, original_time = evaluate_model(original_model, dataloader)
    quantized_accuracy, quantized_time = evaluate_model(quantized_model, dataloader)

    # Print comparisons
    print("Original Model Accuracy:", original_accuracy)
    print("Quantized Model Accuracy:", quantized_accuracy)
    print("Original Model Inference Time:", original_time)
    print("Quantized Model Inference Time:", quantized_time)
    print("Size of Original Model (MB):", os.path.getsize('temp.p') / (1024 * 1024))
    print("Size of Quantized Model (MB):", os.path.getsize('temp_quantized.p') / (1024 * 1024))

def main():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    train_dataset, eval_dataset = load_and_prepare_data(tokenizer)
    model = train_model(model, train_dataset, eval_dataset)

    save_model(model, 'temp.p')
    save_model(model, 'temp_quantized.p', quantized=True)

    compare_models(eval_dataset)

if __name__ == "__main__":
    main()


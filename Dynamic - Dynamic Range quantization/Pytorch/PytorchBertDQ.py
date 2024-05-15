import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def main():
    # Load the dataset and reduce its size for quicker training
    dataset = load_dataset('glue', 'sst2')
    small_train_dataset = dataset['train'].shuffle(seed=42).select(range(500))  # 500 samples for training
    small_eval_dataset = dataset['validation'].shuffle(seed=42).select(range(300))  # 300 samples for validation

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding="max_length", truncation=True)

    # Tokenize the dataset
    tokenized_train_datasets = small_train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_datasets = small_eval_dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,  # Reduced number of epochs
        per_device_train_batch_size=16,  # Adjust batch size if necessary
        per_device_eval_batch_size=32,  # Adjust batch size for evaluation
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_eval_datasets,
    )

    # Train the model
    trainer.train()

    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Evaluate the quantized model
    eval_result = trainer.evaluate(eval_dataset=tokenized_eval_datasets)

    # Print evaluation results and model size comparison
    print("Evaluation results:", eval_result)
    original_model_size = torch.save(model.state_dict(), 'temp.p')
    quantized_model_size = torch.save(quantized_model.state_dict(), 'temp_quantized.p')
    print("Size of original model (MB):", original_model_size / (1024 * 1024))
    print("Size of quantized model (MB):", quantized_model_size / (1024 * 1024))

if __name__ == "__main__":
    main()


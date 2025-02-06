# RoBERTa-based Text Classification with PEFT

## Overview
This project fine-tunes a RoBERTa model for text classification using the AG News dataset. It utilizes Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) to enhance performance while reducing computational costs.

## Dependencies
Ensure you have the required libraries installed:
```bash
pip install transformers datasets evaluate accelerate peft torch tqdm
```

## Dataset
The AG News dataset is used for training, which consists of news articles categorized into four classes:
- World
- Sports
- Business
- Science/Technology

## Model Training
The project implements two training approaches:
1. **Full Fine-Tuning:** The entire RoBERTa model is fine-tuned.
2. **PEFT with LoRA:** A lightweight approach that fine-tunes only a subset of parameters.

### Training Steps
- Tokenize the dataset using `RobertaTokenizer`.
- Fine-tune the RoBERTa model for sequence classification.
- Train using both full fine-tuning and PEFT with LoRA.
- Save the trained models for inference.

## Inference
To classify a text sample using the trained PEFT model:
```python
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer

# Load trained model and tokenizer
inference_model = AutoPeftModelForSequenceClassification.from_pretrained("roberta-base-peft")
tokenizer = AutoTokenizer.from_pretrained("roberta-base-modified")

def classify(text):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    output = inference_model(**inputs)
    prediction = output.logits.argmax(dim=-1).item()
    print(f'Class: {prediction}, Label: {id2label[prediction]}, Text: {text}')

# Example usage
classify("Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.")
```

## Model Evaluation
The trained model is evaluated using accuracy as the primary metric:
```python
import evaluate
from torch.utils.data import DataLoader
from tqdm import tqdm

metric = evaluate.load('accuracy')

def evaluate_model(inference_model, dataset):
    eval_dataloader = DataLoader(dataset.rename_column("label", "labels"), batch_size=8, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_model.to(device)
    inference_model.eval()
    
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = inference_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    eval_metric = metric.compute()
    print(eval_metric)
```

## Results
After training and evaluation, the model achieves a reasonable classification accuracy on the AG News dataset.

## Future Work
- Experiment with different PEFT techniques.
- Optimize training parameters for better performance.
- Deploy the model as an API.

## Author
This project was implemented by **Rediet Solomon**.

## License
This project is licensed under the MIT License.


import torch
import json
import random
import numpy as np
import time
import datetime
import pickle
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# Load intents from JSON file
with open("intents.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)["intents"]

# Extract all unique tags and create a mapping
all_tags = [intent["tag"] for intent in intents_data]
tag_to_responses = {intent["tag"]: intent["responses"] for intent in intents_data}
patterns_to_tags = {pattern.lower(): intent["tag"] for intent in intents_data for pattern in intent["patterns"]}

# Create label encoder
label_encoder = LabelEncoder()
label_encoder.fit(all_tags)

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(all_tags))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train_model():
    texts, tags = zip(*[(pattern, intent["tag"]) for intent in intents_data for pattern in intent["patterns"]])
    encoded_tags = label_encoder.transform(tags)
    encoded_dict = tokenizer.batch_encode_plus(texts, add_special_tokens=True, max_length=128,
                                               padding='max_length', truncation=True, return_tensors='pt')
    input_ids, attention_masks = encoded_dict['input_ids'], encoded_dict['attention_mask']
    labels = torch.tensor(encoded_tags, dtype=torch.long)

    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
        input_ids, attention_masks, labels, random_state=42, test_size=0.2)
    
    batch_size = 32  # Increased batch size
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    epochs = 60  # Increased epochs
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch_i in range(epochs):
        print(f"========== Epoch {epoch_i + 1} / {epochs} ==========")
        model.train()
        total_train_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]
            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss:.4f}")
        
        model.eval()
        total_eval_accuracy, total_eval_loss = 0, 0
        all_preds, all_labels = [], []

        for batch in val_dataloader:
            b_input_ids, b_input_mask, b_labels = [b.to(device) for b in batch]
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                total_eval_loss += outputs.loss.item()
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = b_labels.cpu().numpy()

                preds = np.argmax(logits, axis=1)
                all_preds.extend(preds)
                all_labels.extend(label_ids)

        avg_val_accuracy = accuracy_score(all_labels, all_preds)
        avg_val_f1 = f1_score(all_labels, all_preds, average="weighted")
        avg_val_loss = total_eval_loss / len(val_dataloader)
        print(f"  Validation Accuracy: {avg_val_accuracy:.4f}")
        print(f"  Validation F1 Score: {avg_val_f1:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
    
    print("Training complete!")
    
    model_path = "./pandora_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    with open(f"{model_path}/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

if __name__ == "__main__":
    train_model()
    chat()
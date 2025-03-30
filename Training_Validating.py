import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ===========================
# 1️⃣ Load and Preprocess Data
# ===========================

# Define dataset paths
DATASET_PATH = "dataset/post"  # Update this with your actual dataset folder
LABELS = {"guilty": 1, "innocent": 0}

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Read text files and assign labels
texts, labels = [], []
for label in LABELS:
    folder_path = os.path.join(DATASET_PATH, label)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read().strip())
            labels.append(LABELS[label])

# Train-validation split (80% train, 20% validation)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=None
)

# Tokenize text using BERT tokenizer
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
)
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
)

# Convert labels to tensors
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

# Create DataLoader for training and validation
train_dataset = TensorDataset(
    train_encodings["input_ids"], train_encodings["attention_mask"], train_labels
)
val_dataset = TensorDataset(
    val_encodings["input_ids"], val_encodings["attention_mask"], val_labels
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ===========================
# 2️⃣ Train the BERT Model
# ===========================

# Load BERT model for binary classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Define the scheduler for ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)

# Early Stopping
patience = 10  # Stop after 10 epochs with no improvement
best_val_loss = float("inf")
patience_counter = 0

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Calculate average loss for this epoch
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_train_loss}")

    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Step the scheduler with the validation loss
    scheduler.step(avg_val_loss)
    print("Current LR:", optimizer.param_groups[0]['lr'])

    # Early Stopping: check if validation loss improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "bert_guilty_innocent_best.pth")
        print("Model improved and saved!")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# ===========================
# 3️⃣ Validate the Model
# ===========================

# Load the best saved model if early stopping was triggered
model.load_state_dict(torch.load("bert_guilty_innocent_best.pth"))

model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Final Validation Accuracy: {accuracy:.4f}")
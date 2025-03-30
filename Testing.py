import os
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report

# ===========================
# 1️⃣ Load Model and Tokenizer
# ===========================

# Load trained model and tokenizer
MODEL_PATH = "bert_guilty_innocent_best.pth"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

# ===========================
# 2️⃣ Load Test Data
# ===========================

PREDICTION_PATH = "dataset/pre"
LABELS = {"guilty": 1, "innocent": 0}

test_texts, test_labels, test_filenames = [], [], []

for label_name, label_id in LABELS.items():
    folder_path = os.path.join(PREDICTION_PATH, label_name)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if file_path.endswith(".txt"):  # Ensure only text files are processed
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            
            test_texts.append(text)
            test_labels.append(label_id)
            test_filenames.append(filename)

# ===========================
# 3️⃣ Predict Test Data
# ===========================

def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
    
    return prediction

# Predict for all test samples
predictions = [predict_text(text) for text in test_texts]

# ===========================
# 4️⃣ Evaluate Model
# ===========================

# Compute confusion matrix
cm = confusion_matrix(test_labels, predictions)
print("Confusion Matrix:\n", cm)

# Compute classification report
report = classification_report(test_labels, predictions, target_names=LABELS.keys())
print("\nClassification Report:\n", report)
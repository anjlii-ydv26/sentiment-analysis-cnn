import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import re
from collections import Counter
import pickle

# ----CLEAN----
def clean(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ----LOAD SAVED VOCAB----
print("Loading vocab...")

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# ----ENCODE----
MAX_LEN = 200

def encode(text):
    tokens = text.split()[:MAX_LEN]
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    return ids + [0] * (MAX_LEN - len(ids))

# ----MODEL----
class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()

        EMBEDDING_DIM = 128
        NUM_FILTERS = 150

        self.embedding = nn.Embedding(len(vocab), EMBEDDING_DIM)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, NUM_FILTERS, (k, EMBEDDING_DIM)) for k in [2, 3, 4, 5]
        ])

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(NUM_FILTERS * 4, 2)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)

        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max(i, 2)[0] for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)

        return self.fc(x)

# ----LOAD MODEL----
print("Loading trained model...")

model = TextCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ----PREDICTION----
def predict_sentiment(text):
    text = clean(text)
    encoded = torch.tensor(encode(text)).unsqueeze(0)

    with torch.no_grad():
        output = model(encoded)
        probs = F.softmax(output, dim=1)

        confidence = probs.max().item() * 100
        pred = probs.argmax(1).item()

    # Better decision logic
    if confidence < 60:
        label = "Uncertain"
    else:
        label = "Positive" if pred == 1 else "Negative"

    return label, confidence

# ----RUN LOOP----
print("\nSentiment Analyzer Ready!")
print("Type 'exit' to quit.\n")

while True:
    text = input("Enter review: ")

    if text.lower() == "exit":
        print("Exiting...")
        break

    if len(text.strip()) == 0:
        print("Please enter valid text.\n")
        continue

    label, confidence = predict_sentiment(text)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%\n")
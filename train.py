import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import pickle

# ----CONFIG----
DATA_SIZE = 20000 
EPOCHS = 12
LR = 0.0005
MAX_LEN = 200
VOCAB_SIZE = 30000
BATCH_SIZE = 128
EMBEDDING_DIM = 128
DROPUOUT = 0.5
GRAD_CLIP = 5

# ----CLEAN----
def clean(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# ----LOAD DATA----
print("Loading data...")
df = pd.read_csv("IMDB Dataset.csv").sample(DATA_SIZE, random_state=42)

df["review"] = df["review"].apply(clean)
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

# ----VOCAB----
print("Building vocabulary...")
counter = Counter(" ".join(df["review"]).split())
vocab = {"<pad>": 0, "<unk>": 1}

for word, _ in counter.most_common(VOCAB_SIZE):
    vocab[word] = len(vocab)

with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("Vocab saved!")

# ----ENCODE----
def encode(text):
    tokens = text.split()[:MAX_LEN]
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    return ids + [0] * (MAX_LEN - len(ids))

# -----DATASET----
class IMDBDataset(Dataset):
    def __init__(self, df):
        self.texts = df["review"].tolist()
        self.labels = df["sentiment"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(encode(self.texts[idx])), torch.tensor(self.labels[idx])

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

        self.fc = nn.Linear(NUM_FILTERS * 4, 2)  # 150*4 = 600

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)

        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max(i, 2)[0] for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)

        return self.fc(x)

# ----SPLIT----
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_loader = DataLoader(IMDBDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(IMDBDataset(test_df), batch_size=BATCH_SIZE)

# ----TRAIN----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

losses = []

print("\nTraining Started...\n")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# ----EVALUATE----
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

accuracy = sum(p == t for p, t in zip(y_pred, y_true)) / len(y_true)

print(f"\nAccuracy: {accuracy:.4f}")

# ----SAVE MODEL----
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")

# ----LOG EXPERIMENT----
log_file = "accuracy_log.txt"

with open(log_file, "a") as f:
    f.write(
        f"DATA_SIZE={DATA_SIZE}, "
        f"EPOCHS={EPOCHS}, "
        f"LR={LR}, "
        f"BATCH_SIZE={BATCH_SIZE}, "
        f"ACCURACY={accuracy:.4f}\n"
    )

print("Experiment logged.")

# ----PLOTS----

# Loss Curve
plt.figure()
plt.plot(losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap = "Greens")
plt.title("Confusion Matrix")
plt.show()

# Accuracy Over Runs
if os.path.exists("accuracy_log.txt"):
    accs = []

    with open("accuracy_log.txt", "r") as f:
        for line in f:
            try:
                acc = float(line.strip().split("ACCURACY=")[-1])
                accs.append(acc)
            except:
                continue

    plt.figure()
    plt.plot(accs, marker='o')
    plt.title("Accuracy Over Runs")
    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()
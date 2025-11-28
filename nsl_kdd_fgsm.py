# nsl_kdd_fgsm.py
# Run with: python nsl_kdd_fgsm.py
# Needs:
#   - KDDTrain+.csv
#   - KDDTest+.csv
#   - Field Names.csv  (name,type per line)

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- 1. Load field names ----------
field_df = pd.read_csv("Field Names.csv", header=None, names=["name", "type"])
feature_names = field_df["name"].tolist()
feature_types = field_df["type"].tolist()

column_names = feature_names + ["label", "difficulty"]

categorical_cols = [
    name for name, t in zip(feature_names, feature_types)
    if t.strip().lower() == "symbolic"
]
numerical_cols = [
    name for name, t in zip(feature_names, feature_types)
    if t.strip().lower() == "continuous"
]

# ---------- 2. Load NSL-KDD data ----------
train_df = pd.read_csv("KDDTrain+.csv", header=None)
test_df = pd.read_csv("KDDTest+.csv", header=None)

train_df.columns = column_names
test_df.columns = column_names

# ---------- 3. Labels: normal=0, attack=1 ----------
train_df["binary_label"] = (train_df["label"] != "normal").astype(int)
test_df["binary_label"] = (test_df["label"] != "normal").astype(int)

y_train_full = train_df["binary_label"].values
y_test = test_df["binary_label"].values

train_df = train_df.drop(columns=["label", "difficulty", "binary_label"])
test_df = test_df.drop(columns=["label", "difficulty", "binary_label"])

# ---------- 4. Preprocessing ----------
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numerical_cols),
    ]
)

X_train_full = preprocess.fit_transform(train_df)
X_test = preprocess.transform(test_df)

X_train, X_val_dummy, y_train, y_val_dummy = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# ---------- 5. Torch datasets/loaders ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(x):
    return torch.tensor(
        x.toarray() if hasattr(x, "toarray") else x,
        dtype=torch.float32
    )

X_train_tensor = to_tensor(X_train)
X_test_tensor = to_tensor(X_test)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class NslKddDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    NslKddDataset(X_train_tensor, y_train_tensor),
    batch_size=256,
    shuffle=True,
)
test_loader = DataLoader(
    NslKddDataset(X_test_tensor, y_test_tensor),
    batch_size=512,
    shuffle=False,
)

# ---------- 6. Model ----------
input_dim = X_train_tensor.shape[1]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

model = SimpleMLP(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def accuracy(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == labels).float().mean()

# ---------- 7. Training ----------
print("Training...")
num_epochs = 5

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, y_batch).item()

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    print(f"Epoch {epoch}/{num_epochs}  Loss={avg_loss:.4f}  Acc={avg_acc:.4f}")

# ---------- 8. Baseline evaluation ----------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == y_batch).float().sum().item()
            total += X_batch.size(0)
    return correct / total

clean_acc = evaluate(model, test_loader)
print(f"Clean Test Accuracy: {clean_acc:.4f}")

# ---------- 9. FGSM attack ----------
def fgsm_attack(model, x, y, eps):
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad = True

    logits = model(x_adv)
    loss = criterion(logits, y)
    loss.backward()  # Remove model.zero_grad() line

    grad = x_adv.grad.data.sign()
    return (x_adv + eps * grad).detach()

def evaluate_fgsm(model, loader, eps):
    model.eval()
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        adv = fgsm_attack(model, X_batch, y_batch, eps)
        logits = model(adv)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        correct += (preds == y_batch).float().sum().item()
        total += X_batch.size(0)

    return correct / total

eps = 0.075
adv_acc = evaluate_fgsm(model, test_loader, eps)
print(f"Adversarial Accuracy (FGSM eps={eps}): {adv_acc:.4f}")

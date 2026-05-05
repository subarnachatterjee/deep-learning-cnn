"""
Training and evaluation loop for all 3 CNN models
"""
import torch
import torch.nn as nn
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def train_model(name, model_class, train_loader, test_loader, epochs=10):
    print(f"\nTraining {name} on {DEVICE}...")
    model = model_class().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [],
                "val_loss":   [], "val_acc":   []}

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, test_loader, criterion)
        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(f"  Epoch {epoch+1:02d}/{epochs} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return history

def run_all(model_classes, train_loader, test_loader, epochs=10):
    RESULTS_DIR.mkdir(exist_ok=True)
    all_results = {}

    for name, model_class in model_classes.items():
        history = train_model(name, model_class, train_loader, test_loader, epochs)
        all_results[name] = {
            "best_val_acc": max(history["val_acc"]),
            "final_val_acc": history["val_acc"][-1],
            "history": history
        }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to results/metrics.json")
    return all_results

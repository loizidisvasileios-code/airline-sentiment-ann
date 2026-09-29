"""Batching, class weighting and training."""

import copy  

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader

from src.config import CHECKPOINT_DIR, RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    """Fix the random sources PyTorch uses, so two runs give identical results."""
    torch.manual_seed(seed)  
    np.random.seed(seed)


def _to_dataset(X, y):
    """Pair a feature matrix with its labels as PyTorch tensors."""
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),  # Features must be floats for the linear layers
        torch.tensor(np.asarray(y), dtype=torch.long),  # CrossEntropyLoss expects integer class indices
    )


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=128):
    """Wrap the three splits in DataLoaders that serve the data in batches."""
    train_loader = DataLoader(_to_dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_to_dataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(_to_dataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_class_weights(y_train, device="cpu"):
    """Weight each class according to its frequency in the training set"""
    y = np.asarray(y_train)
    weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    print("Class weights:", weights.round(2))
    return torch.tensor(weights, dtype=torch.float32, device=device)


class EarlyStopping:
    """Stop training once the validation loss has not improved for a number of epochs."""

    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience  # How many epochs of no improvement we tolerate
        self.min_delta = min_delta  # Improvements smaller than this do not count
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """Called once per epoch with the latest validation loss."""
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss  
            self.counter = 0 
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience


def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs=200, patience=5, device="cpu", model_name="model"):
    """Train the model, tracking validation loss every epoch and keeping the best weights."""
    early_stopping = EarlyStopping(patience=patience)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())  

    for epoch in range(num_epochs):
        # ---- Training phase -------------------------------------------------
        model.train() 
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs)  
            loss = criterion(outputs, labels)  
            loss.backward()  # Backpropagation
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)  # Weight by batch size, batches differ in size
        train_loss = running_loss / len(train_loader.dataset)

        # ---- Validation phase -----------------------------------------------
        model.eval()  
        running_loss = 0.0
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                loss = criterion(model(inputs), labels)
                running_loss += loss.item() * inputs.size(0)
        val_loss = running_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1:3d} | Train loss {train_loss:.4f} | Val loss {val_loss:.4f}")

        if val_loss < best_val_loss:  # Remember the best epoch so far
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    model.load_state_dict(best_weights) 
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, CHECKPOINT_DIR / f"{model_name}.pt")
    print(f"Best validation loss {best_val_loss:.4f} | weights saved to checkpoints/{model_name}.pt")
    return model, train_losses, val_losses

"""Controlled comparison of feature representations and class weighting."""

import torch
import torch.nn as nn

from src.evaluate import compute_metrics
from src.model import ANN
from src.train import set_seed, compute_class_weights, train_model


def run_experiment(name, loaders, input_dim, y_train, use_class_weights, lr=1e-4):
    """Train one configuration and measure it on both validation and test."""
    train_loader, val_loader, test_loader = loaders

    set_seed()  

    weights = compute_class_weights(y_train) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)  

    model = ANN(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n=== {name} ===")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, model_name=name
    )

    val_accuracy, val_macro_f1 = compute_metrics(model, val_loader)
    test_accuracy, test_macro_f1 = compute_metrics(model, test_loader)

    return {
        "name": name,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def compare_experiments(results):
    """Print every configuration and return the one with the best validation macro F1."""
    header = f"{'Experiment':<26}{'Val acc':>9}{'Val F1':>9}{'Test acc':>10}{'Test F1':>9}"
    print("\n" + header)
    print("-" * len(header))
    for result in sorted(results, key=lambda r: r["val_macro_f1"], reverse=True):
        print(
            f"{result['name']:<26}{result['val_accuracy']:>9.4f}{result['val_macro_f1']:>9.4f}"
            f"{result['test_accuracy']:>10.4f}{result['test_macro_f1']:>9.4f}"
        )

    
    best = max(results, key=lambda r: r["val_macro_f1"])
    print(f"\nSelected on validation macro F1: {best['name']}")
    return best

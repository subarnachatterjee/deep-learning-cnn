"""
Plot training curves and accuracy comparison
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"

def plot_training_curves(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#4C72B0', '#DD8452', '#55A868']

    for (name, data), color in zip(results.items(), colors):
        epochs = range(1, len(data["history"]["val_acc"]) + 1)
        axes[0].plot(epochs, data["history"]["train_acc"],
                     linestyle='--', color=color, alpha=0.6)
        axes[0].plot(epochs, data["history"]["val_acc"],
                     label=name, color=color)
        axes[1].plot(epochs, data["history"]["val_loss"],
                     label=name, color=color)

    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    print("Saved training_curves.png")

def plot_accuracy_comparison(results):
    names = list(results.keys())
    best_accs  = [results[n]["best_val_acc"] * 100 for n in names]
    final_accs = [results[n]["final_val_acc"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, best_accs,  width, label='Best Val Acc',  color='#4C72B0')
    bars2 = ax.bar(x + width/2, final_accs, width, label='Final Val Acc', color='#55A868')

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CNN Architecture Comparison — Fashion-MNIST")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(70, 100)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_comparison.png", dpi=150)
    print("Saved accuracy_comparison.png")

def generate_all(results):
    plot_training_curves(results)
    plot_accuracy_comparison(results)

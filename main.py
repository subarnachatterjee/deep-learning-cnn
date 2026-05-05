"""
Deep Learning Image Classification Study
=========================================
Benchmarks 3 CNN architectures on Fashion-MNIST:
1. BaseCNN    — simple 2-layer baseline
2. DeepCNN    — deeper 4-layer with BatchNorm
3. ResNetStyle — residual connections

Tracks training curves, validation accuracy and generalisation gap.
"""

from src.data_loader import get_loaders
from src.models import MODELS
from src.trainer import run_all
from src.visualise import generate_all

def main():
    print("=" * 60)
    print("  Deep Learning Image Classification — Fashion-MNIST")
    print("=" * 60)

    # Step 1: Load data with augmentation
    train_loader, test_loader = get_loaders(batch_size=64)

    # Step 2: Train and evaluate all 3 architectures
    results = run_all(MODELS, train_loader, test_loader, epochs=10)

    # Step 3: Generate plots
    generate_all(results)

    # Step 4: Print final summary
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    for name, data in results.items():
        print(f"  {name:<15} Best Val Acc: {data['best_val_acc']*100:.2f}%  "
              f"Final Val Acc: {data['final_val_acc']*100:.2f}%")
    print("=" * 60)
    print("\nDone! Check results/ folder for metrics and plots.")

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_metrics(csv_path):
    df = pd.read_csv(csv_path)
    
    # Extract train and val loss
    train_loss = df[['step', 'train/loss_step']].dropna()
    val_loss = df[['step', 'val/loss']].dropna()
    
    plt.figure(figsize=(10, 6))
    
    # Smooth train loss with rolling window for better visibility
    if len(train_loss) > 10:
        plt.plot(train_loss['step'], train_loss['train/loss_step'], alpha=0.3, color='blue')
        plt.plot(train_loss['step'], train_loss['train/loss_step'].rolling(10).mean(), label='Train Loss (Smoothed)', color='blue')
    else:
        plt.plot(train_loss['step'], train_loss['train/loss_step'], label='Train Loss', color='blue')
        
    plt.plot(val_loss['step'], val_loss['val/loss'], label='Val Loss', color='red', marker='o')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(os.path.dirname(csv_path), 'loss_curve.png')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_metrics(sys.argv[1])
    else:
        print("Usage: python plot_metrics.py path/to/metrics.csv")

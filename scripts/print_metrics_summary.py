import pandas as pd
import sys

def print_summary(csv_path):
    df = pd.read_csv(csv_path)
    
    val_loss = df[['epoch', 'step', 'val/loss']].dropna()
    train_loss = df[['epoch', 'step', 'train/loss_step']].dropna()
    
    print(f"--- Training Summary ---")
    print(f"Total Steps: {df['step'].max()}")
    print(f"Total Epochs: {df['epoch'].max()}")
    
    print("\n--- Validation Loss Timeline ---")
    for _, row in val_loss.iterrows():
        epoch = int(row['epoch'])
        step = int(row['step'])
        loss = row['val/loss']
        
        # Get average train loss around this step
        nearby_train = train_loss[(train_loss['step'] > step - 50) & (train_loss['step'] <= step)]
        avg_train = nearby_train['train/loss_step'].mean() if not nearby_train.empty else float('nan')
        
        print(f"Epoch {epoch:2d} | Step {step:5d} | Val Loss: {loss:.4f} | Train Loss (avg): {avg_train:.4f}")

if __name__ == "__main__":
    print_summary(sys.argv[1])

import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
# Assumes 'train_batch_losses.txt' contains loss for each epoch.
# Assumes 'val_batch_losses.txt' contains loss for every 5th epoch.
TRAIN_LOSS_FILE = 'checkpoints_2/training_loss.txt'
VAL_FREQUENCY = 5  # We have validation loss every 5 epochs.

# Load the loss data from files
try:
    print(f"INFO: Loading training losses from {TRAIN_LOSS_FILE}...")
    train_losses = np.loadtxt(TRAIN_LOSS_FILE)
except FileNotFoundError as e:
    print(f"ERROR: Could not find a loss file. {e}")
    exit()
except Exception as e:
    print(f"ERROR: Could not load loss files. Reason: {e}")
    exit()

# Generate x-axis values (epochs) for each loss type
train_epochs = np.arange(1, len(train_losses) + 1)

# Create the plot
plt.figure(figsize=(12, 7))
plt.plot(train_epochs, train_losses, label='Training Loss (per epoch)', color='royalblue', alpha=0.8)

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs', fontsize=16)
plt.xticks(np.arange(0, len(train_losses) + 1, VAL_FREQUENCY)) # Set x-ticks to be multiples of validation frequency

# Add legend
plt.legend()

# Show the plot
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# If you want to save the plot to a file:
plt.savefig('checkpoints_2/training_validation_loss_plot.png')

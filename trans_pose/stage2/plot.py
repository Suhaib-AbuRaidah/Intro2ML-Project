import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import MaxNLocator

# Load data
df = pd.read_csv("./csvdata/run-.-tag-Loss_train.csv")
df2 = pd.read_csv("./csvdata/run-.-tag-Loss_val_loss.csv")
# df3 = pd.read_csv("./csvdata/run-.-tag-Loss_val_seg_loss.csv")
# df4 = pd.read_csv("./csvdata/run-.-tag-Loss_val_offset_loss.csv")

# Smooth
df["Smoothed"] = gaussian_filter1d(df["Value"], sigma=1)
df2["Smoothed"] = gaussian_filter1d(df2["Value"], sigma=1)
# df3["Smoothed"] = gaussian_filter1d(df3["Value"], sigma=1)
# df4["Smoothed"] = gaussian_filter1d(df4["Value"], sigma=1)

# Convert steps to start at 1
df["Step"] = df["Step"].astype(int) + 1
df2["Step"] = df2["Step"].astype(int) + 1
# df3["Step"] = df3["Step"].astype(int) + 1
# df4["Step"] = df4["Step"].astype(int) + 1

# Plot
plt.figure(figsize=(12,8))
plt.plot(df["Step"], df["Smoothed"], linestyle='-', color="black", label='Total TrainLoss', linewidth=3)
plt.plot(df2["Step"], df2["Smoothed"], linestyle='-', color="blue", label='Total Validation Loss', linewidth=3)
# plt.plot(df3["Step"], df3["Smoothed"], linestyle='--', color="green", label='Seg Loss', linewidth=1)
# plt.plot(df4["Step"], df4["Smoothed"], linestyle='--', color="red", label='Offset Loss', linewidth=1)

# Labels
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("Loss Value", fontsize=20)
plt.title("Total Training and Validation Loss over Epochs", fontsize=26)

# Force integer ticks and start from 1
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # integer ticks only
ax.set_xlim(left=1)  # start x-axis from 1

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("Training_Validation_Loss_over_Epochs.svg")
plt.show()

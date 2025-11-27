import numpy as np
import matplotlib.pyplot as plt

loss_file = "trans_pose/stage2/checkpoints_multi/model2_26nov/loss_history.npz"
data = np.load(loss_file)
train = data['train']
val = data['val']

plt.figure(figsize=(8,5))
plt.plot(train, label='Train Loss')
plt.plot(val, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model 2 Training/Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("model2_loss_curve.png", dpi=120)
plt.show()
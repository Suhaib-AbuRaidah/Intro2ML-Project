# First, save your model to a file
import torch
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)          
        )
    
    def forward(self, x):
        return self.mlp(x)

# Save the model
model = YourModel(in_dim=100, num_classes=10)
dummy_input = torch.randn(1, 100)
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)

print("Model saved as 'model.onnx'. Open with Netron at https://netron.app/")
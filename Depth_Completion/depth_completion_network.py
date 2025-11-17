import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os

INPUT_H, INPUT_W = 360, 640
NUM_INPUT_CHANNELS = 6 

class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthCompletionNet(nn.Module):
    """Simple model: 1 encoder + 1 decoder (FAST)"""
    def __init__(self, input_channels=6, base_channels=32):
        super(DepthCompletionNet, self).__init__()
        
        # Encoder (compress)
        self.enc = nn.Sequential(
            ConvBlock(input_channels, base_channels, 3, 1, 1),
            ConvBlock(base_channels, base_channels * 2, 3, 2, 1),  # /2
            ConvBlock(base_channels * 2, base_channels * 4, 3, 2, 1),  # /4
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, 3, 1, 1),
            ConvBlock(base_channels * 8, base_channels * 4, 3, 1, 1),
        )
        
        # Decoder (decompress)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),  # x2
            ConvBlock(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),  # x2
            ConvBlock(base_channels, base_channels, 3, 1, 1),
            nn.Conv2d(base_channels, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        enc = self.enc(x)
        bottleneck = self.bottleneck(enc)
        out = self.dec(bottleneck)
        return out

# --- Preprocessing: Load and resize on-the-fly ---

def load_and_preprocess_input_fast(path, input_type):
    """Load image, resize to 720x1280, and preprocess (NO SAVING)."""
    if input_type == 'Sparse Depth':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = img / 1000.0  # mm to meters
        img = np.clip(img, 0, 10)
        return torch.from_numpy(img).unsqueeze(0)
    
    elif input_type == 'Normals':
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = 2.0 * img - 1.0
        return torch.from_numpy(img.transpose(2, 0, 1))
    
    elif input_type == 'Mask':
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = img / 255.0
        return torch.from_numpy(img).unsqueeze(0)
    
    else:
        raise ValueError(f"Unknown input type: {input_type}")

def extract_boundaries_from_rgb(rgb_path):
    """Extract boundaries from RGB using Canny edge detection (FAST - no blur)."""
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    
    # Resize
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # FAST Canny (no blur - saves time)
    edges = cv2.Canny(gray, 50, 150)
    edges = edges.astype(np.float32) / 255.0
    return torch.from_numpy(edges).unsqueeze(0)

def load_and_stack_inputs(depth_path, normals_path, mask_path, rgb_path):
    """Load all inputs, resize, extract boundaries, and stack into 6-channel tensor."""
    d = load_and_preprocess_input_fast(depth_path, 'Sparse Depth')
    n = load_and_preprocess_input_fast(normals_path, 'Normals')
    m = load_and_preprocess_input_fast(mask_path, 'Mask')
    b = extract_boundaries_from_rgb(rgb_path)
    
    stacked = torch.cat([d, n, m, b], dim=0)
    return stacked.unsqueeze(0)
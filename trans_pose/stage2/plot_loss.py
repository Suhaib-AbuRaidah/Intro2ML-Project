"""
Extract loss history from saved epoch checkpoints.
Creates loss_history.npz for plotting.
"""
import torch
import numpy as np
import os
from pathlib import Path
import glob
import matplotlib.pyplot as plt

def extract_losses_from_checkpoints(checkpoint_dir):
    """Extract train/val losses from all epoch checkpoints."""
    
    # Find all epoch checkpoints
    pattern = os.path.join(checkpoint_dir, "model_epoch_*.pth")
    ckpt_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if len(ckpt_files) == 0:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return None
    
    print(f"Found {len(ckpt_files)} checkpoints")
    
    train_losses = []
    val_losses = []
    epochs = []
    
    # Also track component losses
    train_pose = []
    train_seg = []
    train_off = []
    val_pose = []
    val_seg = []
    val_off = []
    
    for ckpt_path in ckpt_files:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            epoch = ckpt.get('epoch', len(epochs))
            
            # Total losses (required)
            train_loss = ckpt.get('loss', None)
            val_loss = ckpt.get('val_loss', ckpt.get('best_val_loss', None))
            
            if train_loss is not None:
                epochs.append(epoch + 1)  # 1-indexed
                train_losses.append(train_loss)
                val_losses.append(val_loss if val_loss is not None else train_loss)
                
                # Component losses (optional, may not exist in old checkpoints)
                train_pose.append(ckpt.get('train_pose', 0))
                train_seg.append(ckpt.get('train_seg', 0))
                train_off.append(ckpt.get('train_off', 0))
                val_pose.append(ckpt.get('val_pose', 0))
                val_seg.append(ckpt.get('val_seg', 0))
                val_off.append(ckpt.get('val_off', 0))
            
        except Exception as e:
            print(f"⚠️  Failed to load {ckpt_path}: {e}")
            continue
    
    if len(epochs) == 0:
        print("❌ No valid loss data found in checkpoints")
        return None
    
    # Save to .npz
    save_path = os.path.join(checkpoint_dir, "loss_history.npz")
    np.savez(save_path,
             epochs=np.array(epochs),
             train=np.array(train_losses),
             val=np.array(val_losses),
             train_pose=np.array(train_pose),
             train_seg=np.array(train_seg),
             train_off=np.array(train_off),
             val_pose=np.array(val_pose),
             val_seg=np.array(val_seg),
             val_off=np.array(val_off))
    
    print(f"✅ Saved loss history to {save_path}")
    print(f"   Epochs: {epochs[0]} to {epochs[-1]}")
    print(f"   Train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    print(f"   Val loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")
    
    return save_path


def plot_losses_comprehensive(loss_file, output_dir="loss_plots"):
    """
    Create comprehensive loss visualizations.
    
    Plots:
    1. Total loss (linear + log scale)
    2. Component losses (Pose, Seg, Offset)
    3. Train vs Val comparison
    4. Loss improvement rate
    """
    
    # Load data
    data = np.load(loss_file)
    epochs = data['epochs']
    train = data['train']
    val = data['val']
    
    # Component losses
    train_pose = data.get('train_pose', np.zeros_like(train))
    train_seg = data.get('train_seg', np.zeros_like(train))
    train_off = data.get('train_off', np.zeros_like(train))
    val_pose = data.get('val_pose', np.zeros_like(val))
    val_seg = data.get('val_seg', np.zeros_like(val))
    val_off = data.get('val_off', np.zeros_like(val))
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================
    # PLOT 1: Total Loss (Linear + Log Scale)
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    axes[0].plot(epochs, train, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val, 'r-', label='Val Loss', linewidth=2)
    axes[0].axhline(np.min(val), color='green', linestyle='--', alpha=0.5, label=f'Best Val: {np.min(val):.4f}')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training/Validation Loss (Linear Scale)', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Log scale
    axes[1].plot(epochs, train, 'b-', label='Train Loss', linewidth=2)
    axes[1].plot(epochs, val, 'r-', label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss (log scale)', fontsize=12)
    axes[1].set_title('Training/Validation Loss (Log Scale)', fontsize=14)
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_total_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: 1_total_loss.png")
    
    # ========================================
    # PLOT 2: Component Losses (Train)
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if train_pose.sum() > 0:  # Only plot if data exists
        ax.plot(epochs, train_pose, 'b-', label='Pose Loss', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, train_seg, 'r-', label='Segmentation Loss', linewidth=2, marker='s', markersize=4)
        ax.plot(epochs, train_off, 'g-', label='Offset Loss', linewidth=2, marker='^', markersize=4)
        ax.plot(epochs, train, 'k--', label='Total Loss', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Component Losses', fontsize=14)
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_component_losses_train.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: 2_component_losses_train.png")
    
    # ========================================
    # PLOT 3: Component Losses (Validation)
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if val_pose.sum() > 0:
        ax.plot(epochs, val_pose, 'b-', label='Pose Loss', linewidth=2, marker='o', markersize=4)
        ax.plot(epochs, val_seg, 'r-', label='Segmentation Loss', linewidth=2, marker='s', markersize=4)
        ax.plot(epochs, val_off, 'g-', label='Offset Loss', linewidth=2, marker='^', markersize=4)
        ax.plot(epochs, val, 'k--', label='Total Loss', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Validation Component Losses', fontsize=14)
        ax.set_yscale('log')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_component_losses_val.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: 3_component_losses_val.png")
    
    # ========================================
    # PLOT 4: Train vs Val Gap
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gap = val - train
    ax.plot(epochs, gap, 'purple', linewidth=2, marker='o', markersize=4)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(epochs, 0, gap, where=(gap > 0), color='red', alpha=0.3, label='Overfitting')
    ax.fill_between(epochs, 0, gap, where=(gap < 0), color='green', alpha=0.3, label='Underfitting')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax.set_title('Generalization Gap (Val - Train)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_generalization_gap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: 4_generalization_gap.png")
    
    # ========================================
    # PLOT 5: Loss Improvement Rate
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute relative improvement
    train_improvement = -np.diff(train) / (train[:-1] + 1e-8) * 100  # % improvement
    val_improvement = -np.diff(val) / (val[:-1] + 1e-8) * 100
    
    ax.plot(epochs[1:], train_improvement, 'b-', label='Train Improvement (%)', linewidth=2)
    ax.plot(epochs[1:], val_improvement, 'r-', label='Val Improvement (%)', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss Improvement (%)', fontsize=12)
    ax.set_title('Loss Improvement Rate per Epoch', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_improvement_rate.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: 5_improvement_rate.png")
    
    # ========================================
    # Summary Statistics
    # ========================================
    print("\n" + "="*60)
    print("LOSS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Training Duration: Epoch {epochs[0]} to {epochs[-1]}")
    print(f"\nTrain Loss:")
    print(f"  Initial: {train[0]:.6f}")
    print(f"  Final: {train[-1]:.6f}")
    print(f"  Reduction: {(1 - train[-1]/train[0])*100:.2f}%")
    print(f"\nValidation Loss:")
    print(f"  Initial: {val[0]:.6f}")
    print(f"  Best: {np.min(val):.6f} (Epoch {epochs[np.argmin(val)]})")
    print(f"  Final: {val[-1]:.6f}")
    print(f"  Reduction: {(1 - np.min(val)/val[0])*100:.2f}%")
    print(f"\nOverfitting Check:")
    final_gap = val[-1] - train[-1]
    print(f"  Final Gap (Val - Train): {final_gap:.6f}")
    if final_gap < 0.01:
        print(f"  Status: ✅ Good generalization")
    elif final_gap < 0.05:
        print(f"  Status: ⚠️  Slight overfitting")
    else:
        print(f"  Status: ❌ Significant overfitting")
    print("="*60)



if __name__ == "__main__":
    import argparse
    checkpoint_dir = "checkpoints_multi"
    extract_losses_from_checkpoints(checkpoint_dir)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-file', default='checkpoints_multi/loss_history.npz')
    parser.add_argument('--outdir', default='loss_plots')
    args = parser.parse_args()
    
    if not os.path.exists(args.loss_file):
        print(f"❌ Loss file not found: {args.loss_file}")
        print("   Run extract_loss_history.py first!")
    else:
        plot_losses_comprehensive(args.loss_file, args.outdir)

 
"""
Quick diagnostic: Check data & gradient properties across batches
"""
import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '/content')

from dataset import load_clinical_db
from vmunet_v2 import VM_UNet_V2
from test_dice_functions import build_dice_bce_loss

# Load data
tr_x, tr_y, val_x, val_y, test_x, test_y = load_clinical_db(resize_to=(224,224))
print(f"Data shapes: tr_x={tr_x.shape}, tr_y={tr_y.shape}")
print(f"  Image range: [{tr_x.min():.4f}, {tr_x.max():.4f}]")
print(f"  Label range: [{tr_y.min():.4f}, {tr_y.max():.4f}]")

# Check first few batches
BATCH_SIZE = 4
loss_fn = build_dice_bce_loss()
model = VM_UNet_V2()
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-5, clipnorm=100.0)

print("\n" + "="*70)
print("GRADIENT NORMS ACROSS FIRST 10 BATCHES")
print("="*70)

for batch_num in range(10):
    idx = batch_num * BATCH_SIZE
    if idx + BATCH_SIZE > len(tr_x):
        break
    
    x_batch = tf.cast(tr_x[idx:idx+BATCH_SIZE], tf.float32)
    y_batch = tf.cast(tr_y[idx:idx+BATCH_SIZE], tf.float32)
    
    with tf.GradientTape() as tape:
        y_pred = model(x_batch, training=True)
        loss = loss_fn(y_batch, y_pred)
    
    grads = tape.gradient(loss, model.trainable_weights)
    clipped_grads, grad_norm = tf.clip_by_global_norm(grads, 100.0)
    
    loss_val = float(loss)
    grad_norm_val = float(grad_norm)
    pred_min, pred_max = float(tf.reduce_min(y_pred)), float(tf.reduce_max(y_pred))
    
    print(f"Batch {batch_num+1:2d} | Loss: {loss_val:.4f} | GradNorm: {grad_norm_val:7.4f} | "
          f"Pred: [{pred_min:.4f}, {pred_max:.4f}]")
    
    if grad_norm_val > 100:
        print(f"        ^ ALERT: Gradient explosion!")
        # Debug which layers have large grads
        layer_grads = {}
        for g, w in zip(clipped_grads, model.trainable_weights):
            if g is not None:
                mag = float(tf.reduce_mean(tf.abs(g)))
                layer_grads[w.name] = mag
        
        top_layers = sorted(layer_grads.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"        Top 5 layers with large gradients:")
        for name, mag in top_layers:
            print(f"          - {name}: {mag:.6f}")

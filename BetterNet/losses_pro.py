import tensorflow as tf

def build_boundary_loss():
    """
    Boundary Smoothness Loss (2025 SOTA technique).
    Computes the spatial gradient (edges) of the prediction and the ground truth.
    Penalizes predictions where the edges do not mathematically align with the ground truth edges.
    """
    def compute_gradient(tensor):
        # Sobel filters for edge detection
        kernel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        kernel_x = tf.reshape(kernel_x, [3, 3, 1, 1])
        
        kernel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        kernel_y = tf.reshape(kernel_y, [3, 3, 1, 1])
        
        grad_x = tf.nn.conv2d(tensor, kernel_x, strides=[1, 1, 1, 1], padding='SAME')
        grad_y = tf.nn.conv2d(tensor, kernel_y, strides=[1, 1, 1, 1], padding='SAME')
        
        return tf.sqrt(tf.square(grad_x) + tf.square(grad_y) + 1e-6)

    def loss(y_true, y_pred):
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        true_grad = compute_gradient(y_true)
        pred_grad = compute_gradient(y_pred_clipped)
        
        # Mean Absolute Error of the gradient maps
        boundary_loss = tf.reduce_mean(tf.abs(true_grad - pred_grad))
        return boundary_loss
        
    return loss

def pixel_contrastive_loss(projected_features, masks, temperature=0.1):
    """
    Pixel-Level Contrastive Learning (The Data Efficiency Leap).
    Forces latent feature vectors of 'polyp' pixels to cluster together via Cosine Similarity,
    while repulsing them from 'background' pixels.
    """
    # Downsample the mask to match the deeply projected feature map size (likely 14x14)
    target_shape = tf.shape(projected_features)[1:3]
    resized_masks = tf.image.resize(masks, size=target_shape, method='nearest')
    
    # Flatten spatial dimensions: (Batch, H*W, Channels)
    B = tf.shape(projected_features)[0]
    C = tf.shape(projected_features)[3]
    
    features_flat = tf.reshape(projected_features, [B, -1, C])
    masks_flat = tf.reshape(resized_masks, [B, -1])
    
    # L2 normalize features along channel dimension for cosine similarity
    features_norm = tf.math.l2_normalize(features_flat, axis=-1)
    
    total_loss = 0.0
    valid_batches = 0.0
    
    # Iterate over batch manually (usually small batch size)
    for b in range(B): # tf.range breaks in tf.function without dynamic TensorArray, keeping it simple
        feat = features_norm[b] # (H*W, C)
        mask = tf.cast(masks_flat[b] > 0.5, tf.float32) # (H*W,) Binary
        
        # Identify foreground (polyp) and background pixels
        fg_indices = tf.where(mask > 0)
        bg_indices = tf.where(mask == 0)
        
        # Skip if there's no polyp or no background in this image
        if tf.shape(fg_indices)[0] == 0 or tf.shape(bg_indices)[0] == 0:
            continue
            
        # Sample a few anchor pixels to save memory (Super important for 2025 SOTA)
        num_samples = tf.minimum(tf.shape(fg_indices)[0], 50) # Sample 50 polyp pixels
        fg_sampled_idx = tf.random.shuffle(fg_indices)[:num_samples]
        fg_sampled_idx = tf.reshape(fg_sampled_idx, [-1])
        
        bg_sampled_idx = tf.random.shuffle(bg_indices)[:num_samples]
        bg_sampled_idx = tf.reshape(bg_sampled_idx, [-1])
        
        fg_feats = tf.gather(feat, fg_sampled_idx) # (num_samples, C)
        bg_feats = tf.gather(feat, bg_sampled_idx) # (num_samples, C)
        
        # Compute Cosine Similarities
        # fg_to_fg: Dot product between all sampled polyp pixels
        fg_fg_sim = tf.matmul(fg_feats, fg_feats, transpose_b=True) / temperature
        
        # fg_to_bg: Dot product between polyp and background pixels
        fg_bg_sim = tf.matmul(fg_feats, bg_feats, transpose_b=True) / temperature
        
        # InfoNCE Loss formulation: -log( exp(pos) / (exp(pos) + sum(exp(neg))) )
        # We want to maximize fg_fg instances (positives) and minimize fg_bg (negatives)
        
        # For numerical stability (logsumexp)
        # We consider the diagonal of fg_fg_sim as self-similarity (ignore)
        max_sim = tf.reduce_max(fg_bg_sim, axis=-1, keepdims=True)
        exp_fg_bg = tf.exp(fg_bg_sim - max_sim)
        sum_neg = tf.reduce_sum(exp_fg_bg, axis=-1, keepdims=True)
        
        # Average positive similarities per anchor
        pos_sim = tf.reduce_mean(fg_fg_sim, axis=-1, keepdims=True)
        
        # Contrastive loss per anchor
        anchor_loss = - (pos_sim - max_sim - tf.math.log(sum_neg + 1e-6))
        
        total_loss += tf.reduce_mean(anchor_loss)
        valid_batches += 1.0
        
    final_loss = tf.cond(valid_batches > 0, 
                         lambda: total_loss / valid_batches, 
                         lambda: 0.0)
    return final_loss

def build_betternet_pro_loss(alpha=0.6, beta=0.3, gamma=0.1, contrastive_weight=0.05):
    """
    The Ultimate 100x Loss Factory.
    Combines Dice (primary overlap), BCE (pixel accuracy), Boundary (edge smoothness),
    and Pixel Contrastive (feature clustering).
    """
    boundary_loss_fn = build_boundary_loss()
    
    def loss(y_true, y_pred, projected_features=None):
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # 1. Dice Loss (Alpha)
        smooth = 1e-5
        intersection = tf.reduce_sum(y_true * y_pred_clipped, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_clipped, axis=[1, 2, 3])
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = tf.reduce_mean(dice_loss)
        
        # 2. BCE Loss (Beta)
        bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred_clipped))
        
        # 3. Boundary Loss (Gamma)
        bound_loss = boundary_loss_fn(y_true, y_pred_clipped)
        
        # Core structured loss
        total_structural_loss = (alpha * dice_loss) + (beta * bce_loss) + (gamma * bound_loss)
        
        # 4. Auxiliary Contrastive Loss (Optional, if latent features are passed)
        if projected_features is not None:
            contrast_loss = pixel_contrastive_loss(projected_features, y_true)
            return total_structural_loss + (contrastive_weight * contrast_loss)
            
        return total_structural_loss
        
    return loss

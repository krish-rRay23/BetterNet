import tensorflow as tf

def extract_topological_features(y_true, y_pred, threshold=0.5):
    """
    A Keras-compatible approximation of Topological Data Analysis (TDA).
    Exact Persistent Homology (Betti numbers) is non-differentiable and requires 
    CPU-bound libraries like `giotto-tda` or `GUDHI`, disrupting GPU training graphs.
    
    This function implements a "Differentiable Topological Penalty", focusing on:
    1. Betti 0 (Connected Components): Penalizing fragmented, spurious islands.
    2. Betti 1 (Holes): Penalizing "doughnut" shapes in what should be a solid polyp.
    """
    
    # 1. Soft Thresholding (Differentiable Binarization)
    # y_pred is typically a sigmoid output [0, 1]
    # We want a sharp transition around the threshold without losing gradients.
    steepness = 10.0
    y_pred_bin = tf.math.sigmoid(steepness * (y_pred - threshold))
    
    # We apply Morphological Operations (Erosion and Dilation) to detect topology.
    # In continuous space, these are approximated by MinPooling and MaxPooling.
    kernel_size = 5
    
    # 2. Approximate Betti-0 Penalty (Fragmentation)
    # If we dilate the prediction, fragments merge. 
    # If the dilated volume is significantly larger than the original, it means the prediction was highly fragmented.
    y_pred_dilated = tf.nn.max_pool2d(y_pred_bin, ksize=kernel_size, strides=1, padding='SAME')
    fragmentation_penalty = tf.reduce_mean(y_pred_dilated - y_pred_bin)
    
    # 3. Approximate Betti-1 Penalty (Holes)
    # If we erode the prediction, thin borders disappear.
    # If the eroded mask loses more volume than expected (compared to the ground truth eroded), 
    # it implies the prediction has a "thin shell" or holes.
    y_pred_eroded = -tf.nn.max_pool2d(-y_pred_bin, ksize=kernel_size, strides=1, padding='SAME')
    y_true_eroded = -tf.nn.max_pool2d(-y_true, ksize=kernel_size, strides=1, padding='SAME')
    
    # We penalize if the network's eroded core is smaller than the true eroded core.
    hole_penalty = tf.reduce_mean(tf.nn.relu(y_true_eroded - y_pred_eroded))

    return fragmentation_penalty, hole_penalty

def topological_loss(weight_frag=0.1, weight_hole=0.1):
    """
    Returns a custom Keras loss function that combines standard BCE/Dice 
    with the differentiable Topological Data Analysis (TDA) penalties.
    """
    def loss(y_true, y_pred):
        # Base Loss: Binary Cross-Entropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Base Loss: Dice
        smooth = 1e-5
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = tf.reduce_mean(dice_loss)

        # Topological Penalties (TDA Approximation)
        frag_penalty, hole_penalty = extract_topological_features(y_true, y_pred)
        
        # Total Loss
        total_loss = bce + dice_loss + (weight_frag * frag_penalty) + (weight_hole * hole_penalty)
        return total_loss
        
    return loss

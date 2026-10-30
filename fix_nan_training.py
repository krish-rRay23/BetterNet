import nbformat

def fix_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == 'code' and 'def train_step(x, y):' in cell.source:
            
            old_code = """        # === COMPILED TRAINING STEP ===
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = loss_fn(y, y_pred)
                
            grads = tape.gradient(loss, model.trainable_weights)
            clipped_grads, grad_norm = tf.clip_by_global_norm(grads, GRAD_CLIP_NORM)
            optimizer.apply_gradients(zip(clipped_grads, model.trainable_weights))
            
            dice = dice_coef(y, y_pred)
            return loss, dice, grad_norm"""
            
            new_code = """        # === COMPILED TRAINING STEP ===
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss = loss_fn(y, y_pred)
                
            grads = tape.gradient(loss, model.trainable_weights)
            
            # Safely replace NaNs in gradients with 0s just in case, though is_finite protects us
            safe_grads = [tf.where(tf.math.is_nan(g), tf.zeros_like(g), g) if g is not None else None for g in grads]
            clipped_grads, grad_norm = tf.clip_by_global_norm(safe_grads, GRAD_CLIP_NORM)
            
            # Only apply gradients if both the loss and the gradients are finite
            is_loss_finite = tf.math.is_finite(loss)
            is_grad_finite = tf.math.is_finite(grad_norm)
            is_safe = tf.logical_and(is_loss_finite, is_grad_finite)
            
            if is_safe:
                optimizer.apply_gradients(zip(clipped_grads, model.trainable_weights))
                
            dice = dice_coef(y, y_pred)
            return loss, dice, grad_norm"""
            
            if 'optimizer.apply_gradients(zip(clipped_grads, model.trainable_weights))' in cell.source:
                cell.source = cell.source.replace('optimizer.apply_gradients(zip(clipped_grads, model.trainable_weights))', 'is_loss_finite = tf.math.is_finite(loss)\n            is_grad_finite = tf.math.is_finite(grad_norm)\n            is_safe = tf.logical_and(is_loss_finite, is_grad_finite)\n            if is_safe:\n                optimizer.apply_gradients(zip(clipped_grads, model.trainable_weights))')

    with open(file_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

fix_notebook('train_minimal_working.ipynb')
print("Patched train_step in train_minimal_working.ipynb")

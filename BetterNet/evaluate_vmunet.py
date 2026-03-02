import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

def calculate_metrics(y_true, y_pred):
    """
    Calculates a rigorous suite of evaluation metrics for medical image segmentation.
    This ensures enterprise-level benchmarking for the 100x Leap model.
    """
    # Flatten arrays
    y_true_f = np.round(y_true.flatten())
    y_pred_f = np.round(y_pred.flatten())
    
    # Core Metrics
    acc = accuracy_score(y_true_f, y_pred_f)
    f1 = f1_score(y_true_f, y_pred_f) # Dice equivalent
    jac = jaccard_score(y_true_f, y_pred_f) # IoU equivalent
    recall = recall_score(y_true_f, y_pred_f) # Sensitivity
    precision = precision_score(y_true_f, y_pred_f) # PPV
    
    return acc, f1, jac, recall, precision

def evaluate_model_comprehensive(model, dataset, num_samples_to_visualize=5):
    """
    Runs the dataset through the model, computes average metrics,
    and generates visual saliency maps comparing GT to Prediction.
    """
    print("Beginning rigorous evaluation over validation dataset...")
    
    metrics_log = []
    
    plt.figure(figsize=(15, 5 * num_samples_to_visualize))
    visuals_plotted = 0
    
    for batch_id, (imgs, masks) in enumerate(dataset):
        preds = model.predict(imgs, verbose=0)
        
        # Calculate metrics for the batch
        for i in range(len(imgs)):
            y_t = masks[i].numpy()
            y_p = preds[i] > 0.5
            
            acc, f1, jac, recall, precision = calculate_metrics(y_t, y_p)
            metrics_log.append([acc, f1, jac, recall, precision])
            
            # Plot visuals if needed
            if visuals_plotted < num_samples_to_visualize:
                plt.subplot(num_samples_to_visualize, 3, visuals_plotted*3 + 1)
                plt.title("Input Image")
                plt.imshow(imgs[i])
                plt.axis('off')
                
                plt.subplot(num_samples_to_visualize, 3, visuals_plotted*3 + 2)
                plt.title("Ground Truth Mask")
                plt.imshow(y_t[..., 0], cmap='gray')
                plt.axis('off')
                
                plt.subplot(num_samples_to_visualize, 3, visuals_plotted*3 + 3)
                plt.title(f"VM-UNet V2 Prediction (Dice: {f1:.3f})")
                plt.imshow(y_p[..., 0], cmap='gray')
                plt.axis('off')
                
                visuals_plotted += 1

    # Print Tabular Summary
    metrics_array = np.array(metrics_log)
    means = np.mean(metrics_array, axis=0)
    
    print("\n--- VM-UNet V2 Evaluation Results ---")
    print(f"Accuracy:  {means[0]:.4f}")
    print(f"F1 (Dice): {means[1]:.4f}")
    print(f"Jaccard:   {means[2]:.4f}")
    print(f"Recall:    {means[3]:.4f}")
    print(f"Precision: {means[4]:.4f}")
    print("-------------------------------------")
    
    plt.tight_layout()
    plt.show()

# Example usage (assuming 'model' and 'val_dataset' exist in scope):
# evaluate_model_comprehensive(model, val_dataset)

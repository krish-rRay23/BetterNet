import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Activation, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3

try:
    from layers import CLFFModule
except (ImportError, ModuleNotFoundError):
    from BetterNet.layers import CLFFModule

def betternet_pro(input_shape=(224, 224, 3), num_classes=1, base_filters=32):
    """
    BetterNet-Pro (2025 SOTA Edition).
    Utilitizes EfficientNet-B3 backbone and Cross-Level Feature Fusion (CLFF) for explicit boundary awareness.
    """
    inputs = Input(shape=input_shape, name="input_image")
    
    # --- 1. Encoder (EfficientNet-B3 Backbone) ---
    # We use a pre-trained backbone as it's proven vastly superior to training from scratch on small medical datasets.
    encoder = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Extract hierarchical features
    # Note: These layer names are specific to EfficientNetB3
    s1 = encoder.get_layer("input_image").output                           # (224, 224, 3)
    s2 = encoder.get_layer("block2a_expand_activation").output             # (112, 112, 144)
    s3 = encoder.get_layer("block3a_expand_activation").output             # (56, 56, 192)
    s4 = encoder.get_layer("block4a_expand_activation").output             # (28, 28, 288)
    
    # Bottleneck
    s5 = encoder.get_layer("block6a_expand_activation").output             # (14, 14, 816)
    
    # Deepest projection for Pixel Contrastive Learning
    # We expose this vector out of the model so the custom training loop can calculate the contrastive loss.
    projected_s5 = Conv2D(64, (1, 1), name="contrastive_projection")(s5)

    # --- 2. Decoder with CLFF ---
    
    # Decode Stage 4
    d4 = UpSampling2D((2, 2), interpolation='bilinear')(s5)
    d4 = CLFFModule(filters=base_filters * 8)([s4, d4])
    d4 = Dropout(0.3)(d4)
    
    # Decode Stage 3
    d3 = UpSampling2D((2, 2), interpolation='bilinear')(d4)
    d3 = CLFFModule(filters=base_filters * 4)([s3, d3])
    d3 = Dropout(0.2)(d3)
    
    # Decode Stage 2
    d2 = UpSampling2D((2, 2), interpolation='bilinear')(d3)
    d2 = CLFFModule(filters=base_filters * 2)([s2, d2])
    d2 = Dropout(0.1)(d2)
    
    # Decode Stage 1
    d1 = UpSampling2D((2, 2), interpolation='bilinear')(d2)
    d1 = CLFFModule(filters=base_filters)([s1, d1])
    
    # --- 3. Final Prediction Head ---
    out = Conv2D(base_filters, (3, 3), padding='same', activation='relu')(d1)
    out = BatchNormalization()(out)
    out = Conv2D(num_classes, (1, 1), padding='same')(out)
    
    # Critical: Outputs valid probabilities
    output = Activation("sigmoid", name="output_image")(out)
    
    # We return both the main output and the latent contrastive projection
    model = Model(inputs, outputs=[output, projected_s5], name="BetterNet_Pro")
    return model

if __name__ == "__main__":
    # Test Compilation
    model = betternet_pro()
    model.summary()

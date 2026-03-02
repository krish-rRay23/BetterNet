import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Activation, Resizing, SpatialDropout2D
from layers import CBAMModule
from freq_mamba import DualGateFrequencyModule
from mamba import VSSBlock
from tensorflow.keras.models import Model


class SDIModule(tf.keras.layers.Layer):
    """
    Semantics and Detail Infusion (SDI) Module.
    Fuses hierarchical features from the encoder using the CBAM attention mechanism.
    """
    def __init__(self, filters, **kwargs):
        super(SDIModule, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv_fuse = Conv2D(self.filters, (3, 3), padding='same', activation='relu')
        self.cbam = CBAMModule(ratio=8, kernel_size=7)
        super(SDIModule, self).build(input_shape)

    def call(self, encoder_feat, decoder_feat):
        target_hw = tf.shape(encoder_feat)[1:3]
        decoder_feat_resized = tf.image.resize(decoder_feat, size=target_hw, method='bilinear')
        fused = Concatenate()([encoder_feat, decoder_feat_resized])
        fused = self.conv_fuse(fused)
        fused = self.cbam(fused)
        return fused

def vmunet_v2(input_shape=(256, 256, 3), num_classes=1, base_filters=32, use_freq_mamba=True, lightweight_stem=True):
    """
    VM-UNet V2 Architecture (Vision Mamba U-Net).
    Includes optional End-Stage Frequency Enhancement (`use_freq_mamba`).
    """
    inputs = Input(shape=input_shape, name="input_image")
    
    # --- Encoder (VSS Blocks) ---
    x1 = Conv2D(base_filters, (3, 3), padding='same', strides=2, activation='relu')(inputs)
    if not lightweight_stem:
        x1 = VSSBlock(d_model=base_filters)(x1)
    x1 = SpatialDropout2D(0.2)(x1)
    
    x2 = Conv2D(base_filters * 2, (3, 3), padding='same', strides=2)(x1)
    x2 = VSSBlock(d_model=base_filters * 2)(x2)
    x2 = SpatialDropout2D(0.2)(x2)
    
    x3 = Conv2D(base_filters * 4, (3, 3), padding='same', strides=2)(x2)
    x3 = VSSBlock(d_model=base_filters * 4)(x3)
    x3 = SpatialDropout2D(0.2)(x3)
    
    # Stage 4 (Bottleneck)
    x4 = Conv2D(base_filters * 8, (3, 3), padding='same', strides=2)(x3)
    x4 = VSSBlock(d_model=base_filters * 8)(x4)
    x4 = SpatialDropout2D(0.3)(x4)
    x4_out = VSSBlock(d_model=base_filters * 8)(x4)
    x4_out = SpatialDropout2D(0.3)(x4_out)

    # --- Decoder (SDI Modules) ---
    d3 = UpSampling2D((2, 2), interpolation='bilinear')(x4_out)
    d3 = Conv2D(base_filters * 4, (1, 1), padding='same')(d3)
    sdi_3 = SDIModule(filters=base_filters * 4)(x3, d3)
    
    d2 = UpSampling2D((2, 2), interpolation='bilinear')(sdi_3)
    d2 = Conv2D(base_filters * 2, (1, 1), padding='same')(d2)
    sdi_2 = SDIModule(filters=base_filters * 2)(x2, d2)
    
    d1 = UpSampling2D((2, 2), interpolation='bilinear')(sdi_2)
    d1 = Conv2D(base_filters, (1, 1), padding='same')(d1)
    sdi_1 = SDIModule(filters=base_filters)(x1, d1)
    
    out = UpSampling2D((2, 2), interpolation='bilinear')(sdi_1)
    out = Conv2D(base_filters, (3, 3), padding='same')(out)

    # --- 100x Leap: Dual-Gate Frequency Enhancement ---
    # We apply FFT logic at the final high-resolution stage before the segmentation head
    # so the model can sharply refine the polyp boundaries in frequency space.
    if use_freq_mamba:
        out = DualGateFrequencyModule(d_model=base_filters)(out)
    
    # Final Segmentation Head
    out = Conv2D(num_classes, (1, 1), padding='same')(out)
    output = Activation("sigmoid", name="output_image")(out)
    
    model = Model(inputs, outputs=output, name="VMUNet_V2_FreqMamba" if use_freq_mamba else "VMUNet_V2")
    return model
    
if __name__ == "__main__":
    # Test compilation
    model = vmunet_v2(input_shape=(256, 256, 3), use_freq_mamba=True)
    model.summary()


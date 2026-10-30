import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Activation, GlobalAveragePooling2D,
                                     GlobalMaxPooling2D, Dense, Add, Concatenate, multiply, Layer)

class ChannelAttentionModule(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttentionModule, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_dense_one = Dense(channel // self.ratio, 
                                      activation='relu', 
                                      kernel_initializer='he_normal', 
                                      use_bias=True, 
                                      bias_initializer='zeros')
        self.shared_dense_two = Dense(channel, 
                                      kernel_initializer='he_normal', 
                                      use_bias=True, 
                                      bias_initializer='zeros')
        super(ChannelAttentionModule, self).build(input_shape)

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D()(inputs)    
        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = GlobalMaxPooling2D()(inputs)
        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        attention_feature = Add()([avg_pool, max_pool])
        attention_feature = Activation('sigmoid')(attention_feature)
        attention_feature = tf.reshape(attention_feature, [-1, 1, 1, tf.shape(attention_feature)[-1]])

        return multiply([inputs, attention_feature])

    def get_config(self):
        config = super(ChannelAttentionModule, self).get_config()
        config.update({'ratio': self.ratio})
        return config

class SpatialAttentionModule(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttentionModule, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv2d = Conv2D(filters=1, 
                             kernel_size=self.kernel_size, 
                             strides=1, 
                             padding='same', 
                             activation='sigmoid', 
                             kernel_initializer='he_normal', 
                             use_bias=False)
        super(SpatialAttentionModule, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        attention_feature = self.conv2d(concat)   

        return multiply([inputs, attention_feature])

    def get_config(self):
        config = super(SpatialAttentionModule, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

class CBAMModule(Layer):
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super(CBAMModule, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttentionModule(ratio=ratio)
        self.spatial_attention = SpatialAttentionModule(kernel_size=kernel_size)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super(CBAMModule, self).get_config()
        config.update({
            'ratio': self.ratio,
            'kernel_size': self.kernel_size
        })
        return config

class CLFFModule(Layer):
    """
    Cross-Level Feature Fusion (CLFF) Module.
    Computes a spatial differential between high-resolution and low-resolution features
    to explicitly guide attention toward boundaries (2025 SOTA technique).
    """
    def __init__(self, filters, **kwargs):
        super(CLFFModule, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        # input_shape is a list: [high_res_shape, low_res_shape]
        self.conv_high = Conv2D(self.filters, (1, 1), padding='same', use_bias=False)
        self.conv_low = Conv2D(self.filters, (1, 1), padding='same', use_bias=False)
        
        self.conv_fusion = Conv2D(self.filters, (3, 3), padding='same', activation='relu')
        self.cbam = CBAMModule(ratio=8, kernel_size=7)
        super(CLFFModule, self).build(input_shape)

    def call(self, inputs):
        # inputs[0]: High-resolution feature map (from encoder)
        # inputs[1]: Low-resolution feature map (from decoder)
        high_res, low_res = inputs
        
        # 1. Align dimensions
        target_hw = tf.shape(high_res)[1:3]
        low_res_up = tf.image.resize(low_res, size=target_hw, method='bilinear')
        
        # 2. Project to same channel depth
        h_proj = self.conv_high(high_res)
        l_proj = self.conv_low(low_res_up)
        
        # 3. Compute Boundary Map (Differential)
        # Absolute difference highlights where the high-res details disagree with low-res semantics (the boundaries)
        diff_map = tf.abs(h_proj - l_proj)
        
        # 4. Modulate the original features with the boundary map
        fused = Concatenate()([h_proj, l_proj, diff_map])
        fused = self.conv_fusion(fused)
        
        # 5. Apply CBAM attention to the boundary-aware fused feature
        out = self.cbam(fused)
        
        return out
        
    def get_config(self):
        config = super(CLFFModule, self).get_config()
        config.update({'filters': self.filters})
        return config

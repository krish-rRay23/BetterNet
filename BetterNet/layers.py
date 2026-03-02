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

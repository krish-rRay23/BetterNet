import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Activation, multiply, Add, LayerNormalization

class FFTLayer(Layer):
    """
    A Keras Layer that computes the 2D Fast Fourier Transform (FFT).
    It extracts the amplitude and phase components of the spatial image.
    """
    def __init__(self, **kwargs):
        super(FFTLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Convert to complex64 for FFT processing
        x_complex = tf.cast(inputs, tf.complex64)
        
        # Compute 2D FFT
        # tf.signal.fft2d computes the 2D discrete Fourier transform over the inner-most 2 dimensions
        # Inputs shape: (Batch, Height, Width, Channels) => We transpose to compute FFT spatially
        x_transposed = tf.transpose(x_complex, perm=[0, 3, 1, 2]) # (B, C, H, W)
        fft_out = tf.signal.fft2d(x_transposed)
        
        # Shift the zero-frequency component to the center of the spectrum
        fft_shifted = tf.signal.fftshift(fft_out, axes=(-2, -1))
        
        # Extract Amplitude and Phase
        amplitude = tf.abs(fft_shifted)
        phase = tf.math.angle(fft_shifted)
        
        # Transpose back to standard Keras format (B, H, W, C)
        amplitude = tf.transpose(amplitude, perm=[0, 2, 3, 1])
        phase = tf.transpose(phase, perm=[0, 2, 3, 1])
        
        return amplitude, phase, fft_shifted

class InverseFFTLayer(Layer):
    """
    Computes the Inverse 2D FFT to convert frequency domain features back to spatial domain.
    """
    def __init__(self, **kwargs):
        super(InverseFFTLayer, self).__init__(**kwargs)

    def call(self, amplitude, phase, original_fft_shifted=None):
        # Store original dtype for casting back later
        original_dtype = amplitude.dtype
        
        # Cast to float32 for complex operations (tf.complex requires float32/float64)
        amplitude = tf.cast(amplitude, tf.float32)
        phase = tf.cast(phase, tf.float32)
        
        # Reconstruct complex tensor from modified amplitude and phase
        complex_reconstructed = tf.complex(amplitude * tf.cos(phase), amplitude * tf.sin(phase))
        
        # If we passed the original shifted FFT, we just inverse shift it
        if original_fft_shifted is not None:
             # Use the modified complex reconstructed tensor, but we need to unshift it first
             fft_unshifted = tf.signal.ifftshift(complex_reconstructed, axes=(1, 2))
             # Inverse FFT expects (B, C, H, W), so we transpose
             fft_unshifted = tf.transpose(fft_unshifted, perm=[0, 3, 1, 2])
        else:
             fft_unshifted = tf.transpose(complex_reconstructed, perm=[0, 3, 1, 2])
             fft_unshifted = tf.signal.ifftshift(fft_unshifted, axes=(2, 3))
             
        # Inverse 2D FFT
        spatial_out = tf.signal.ifft2d(fft_unshifted)
        spatial_out = tf.math.real(spatial_out) # We only care about the real part in image land
        
        # Transpose back to (B, H, W, C)
        spatial_out = tf.transpose(spatial_out, perm=[0, 2, 3, 1])
        # Return in same dtype as input amplitude (preserves FP16 policy)
        return tf.cast(spatial_out, original_dtype)

class DualGateFrequencyModule(Layer):
    """
    FreqMamba's core innovation. It processes features in the frequency domain,
    using gating mechanisms to explicitly enhance high-frequency details (polyp edges)
    that standard Mamba blocks usually suppress.
    """
    def __init__(self, d_model, **kwargs):
        super(DualGateFrequencyModule, self).__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.fft = FFTLayer()
        self.ifft = InverseFFTLayer()
        
        # High/Low frequency gating convolutions applied to the amplitude spectrum
        self.high_freq_gate = Conv2D(self.d_model, (1, 1), padding='same', activation='sigmoid')
        self.low_freq_gate = Conv2D(self.d_model, (1, 1), padding='same', activation='sigmoid')
        
        # Feature fusion after inverse FFT
        self.fusion_conv = Conv2D(self.d_model, (3, 3), padding='same', activation='swish')
        self.norm = LayerNormalization()
        super(DualGateFrequencyModule, self).build(input_shape)

    def call(self, inputs):
        # 1. Spatial to Frequency Domain
        amplitude, phase, fft_shifted = self.fft(inputs)
        
        # 2. Dual-Gate Frequency Modulation
        # We learn distinct gates to enhance crucial frequencies. 
        # Typically, boundary details reside in higher frequencies.
        gate_h = self.high_freq_gate(amplitude)
        gate_l = self.low_freq_gate(amplitude)
        
        # Combine gates to modulate the amplitude
        modulated_amplitude = multiply([amplitude, gate_h]) + multiply([amplitude, gate_l])
        
        # 3. Frequency to Spatial Domain
        # We perform IFFT using the enhanced amplitude and the original phase
        spatial_enhanced = self.ifft(modulated_amplitude, phase, fft_shifted)
        
        # 4. Residual Fusion
        fused = self.fusion_conv(spatial_enhanced)
        fused = self.norm(fused + inputs) # Add residual spatial connection
        
        return fused

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, SeparableConv1D, Dense, LayerNormalization, Activation
import math

class MambaLayer(Layer):
    """
    A 1D Mamba block implementation in Keras.
    This approximates the continuous-time sequence modeling of State Space Models (SSMs).
    """
    def __init__(self, d_model, d_state=8, d_conv=4, expand=2, dt_rank="auto", **kwargs):
        super(MambaLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

    def build(self, input_shape):
        # Input projection
        self.in_proj = Dense(self.d_inner * 2, use_bias=False)
        
        # 1D Convolution
        self.conv1d = SeparableConv1D(filters=self.d_inner, kernel_size=self.d_conv,
                          padding="causal", activation="swish")
        
        # SSM parameters (x_proj handles dt, B, C)
        self.x_proj = Dense(self.dt_rank + self.d_state * 2, use_bias=False)
        self.dt_proj = Dense(self.d_inner, use_bias=True)
        
        # A matrix (transition parameters) - initialized to be stable
        A = tf.repeat(tf.range(1, self.d_state + 1, dtype=tf.float32), self.d_inner)
        self.A_log = tf.Variable(tf.math.log(tf.reshape(A, (self.d_inner, self.d_state))), trainable=True, name="A_log")
        
        # D feedforward parameter
        self.D = tf.Variable(tf.ones(self.d_inner), trainable=True, name="D")
        
        # Output projection
        self.out_proj = Dense(self.d_model, use_bias=False)
        super(MambaLayer, self).build(input_shape)

    def call(self, x):
        """ x shape: (batch_size, seq_len, d_model) """
        # Cast SSM parameters to match input dtype (FP16 compatibility)
        compute_dtype = x.dtype
        
        # 1. Project input -> shape: (batch, seq, 2 * d_inner)
        xz = self.in_proj(x)
        x_in, z = tf.split(xz, 2, axis=-1) 

        # 2. 1D Convolution -> shape: (batch, seq, d_inner)
        x_conv = self.conv1d(x_in)

        # 3. Dynamic projections (B, C, delta)
        x_dbl = self.x_proj(x_conv) # shape: (batch, seq, dt_rank + 2*d_state)
        delta, B, C = tf.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], axis=-1)
        
        # Delta projection (dt)
        delta = self.dt_proj(delta) # shape: (batch, seq, d_inner)
        delta = tf.math.softplus(delta) # Ensure dt is positive

        # 4. Selective Scan (Discretization & Recurrence)
        # We use a custom tf.scan to simulate the SSM recurrence over the sequence.
        # This is the O(N) mechanism that replaces self-attention.
        A = -tf.exp(tf.cast(self.A_log, compute_dtype)) # shape: (d_inner, d_state)
        
        # Discretize A and B
        # delta shape: (B, L, D), A shape: (D, N) -> deltaA : (B, L, D, N)
        deltaA = tf.exp(tf.einsum('bld,dn->bldn', delta, A))
        # B shape: (B, L, N), delta shape: (B, L, D), x_conv shape: (B, L, D) -> deltaB_u: (B, L, D, N)
        deltaB_u = tf.einsum('bld,bln,bld->bldn', delta, B, x_conv)

        # Recurrence: h_t = (deltaA * h_{t-1}) + deltaB_u
        def scan_fn(h_prev, current_inputs):
            dA_t, dB_u_t = current_inputs
            return dA_t * h_prev + dB_u_t
            
        # Initial hidden state: shape (batch, d_inner, d_state)
        h_0 = tf.zeros((tf.shape(x)[0], self.d_inner, self.d_state), dtype=compute_dtype)
        
        # Apply sequential scan along sequence length (L) using tf.scan
        # We need to unstack along sequence dim to feed into scan
        # Note: tf.scan processes shape (L, B, D, N), so we transpose inputs first
        dA_seq = tf.transpose(deltaA, perm=[1, 0, 2, 3])
        dB_u_seq = tf.transpose(deltaB_u, perm=[1, 0, 2, 3])
        
        # h shape: (L, B, D, N)
        h = tf.scan(scan_fn, elems=(dA_seq, dB_u_seq), initializer=h_0)
        
        # Back to original shape (B, L, D, N)
        h = tf.transpose(h, perm=[1, 0, 2, 3])
        
        # Output computation: y_t = C * h_t + D * u_t
        # C shape: (B, L, N), h shape: (B, L, D, N) -> y: (B, L, D)
        y = tf.einsum('bln,bldn->bld', C, h)
        y = y + x_conv * tf.cast(self.D, compute_dtype)
        
        # 5. Gating and output projection
        y = y * tf.nn.silu(z)
        out = self.out_proj(y)
        
        return out


class SS2D(Layer):
    """
    2D Selective Scan (SS2D) module.
    Flattens 2D features and processes them in 4 directions through 1D Mamba blocks.
    """
    def __init__(self, d_model, **kwargs):
        super(SS2D, self).__init__(**kwargs)
        self.d_model = d_model
        
    def build(self, input_shape):
        self.mamba_forward = MambaLayer(self.d_model)
        self.mamba_backward = MambaLayer(self.d_model)
        self.mamba_forward_t = MambaLayer(self.d_model)
        self.mamba_backward_t = MambaLayer(self.d_model)
        self.out_proj = Dense(self.d_model)
        super(SS2D, self).build(input_shape)

    def call(self, x):
        batch, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # 1. Forward scan (Top-Left to Bottom-Right)
        x_flatten = tf.reshape(x, (batch, h * w, c))
        out_f = self.mamba_forward(x_flatten)
        
        # 2. Backward scan (Bottom-Right to Top-Left)
        x_flatten_b = tf.reverse(x_flatten, axis=[1])
        out_b = self.mamba_backward(x_flatten_b)
        out_b = tf.reverse(out_b, axis=[1])
        
        # 3. Forward transposed scan (Top-Right to Bottom-Left)
        x_t = tf.transpose(x, perm=[0, 2, 1, 3]) # transpose spatial dims
        x_flatten_t = tf.reshape(x_t, (batch, w * h, c))
        out_f_t = self.mamba_forward_t(x_flatten_t)
        out_f_t = tf.reshape(out_f_t, (batch, w, h, c))
        out_f_t = tf.transpose(out_f_t, perm=[0, 2, 1, 3]) # transpose back
        
        # 4. Backward transposed scan (Bottom-Left to Top-Right)
        x_flatten_t_b = tf.reverse(x_flatten_t, axis=[1])
        out_b_t = self.mamba_backward_t(x_flatten_t_b)
        out_b_t = tf.reverse(out_b_t, axis=[1])
        out_b_t = tf.reshape(out_b_t, (batch, w, h, c))
        out_b_t = tf.transpose(out_b_t, perm=[0, 2, 1, 3]) # transpose back
        
        # Merge the 4 directional scans
        out_f = tf.reshape(out_f, (batch, h, w, c))
        out_b = tf.reshape(out_b, (batch, h, w, c))
        
        merged = out_f + out_b + out_f_t + out_b_t
        return self.out_proj(merged)

class VSSBlock(Layer):
    """
    Visual State Space (VSS) Block used in VM-UNet V2.
    """
    def __init__(self, d_model, **kwargs):
        super(VSSBlock, self).__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.ln1 = LayerNormalization()
        self.ss2d = SS2D(self.d_model)
        self.ln2 = LayerNormalization()
        # Optional FFN could be placed here if needed
        super(VSSBlock, self).build(input_shape)

    def call(self, x):
        # Residual connection
        return x + self.ss2d(self.ln1(x))

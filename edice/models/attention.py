"""Implementations of single-headed attention and multi head attention.

Based on tf documentation:
https://www.tensorflow.org/tutorials/text/transformer#encoder_layer
"""
import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """
    Taken from tf tutorial https://www.tensorflow.org/tutorials/text/transformer#encoder_layer
    Mask should be 1 if you're masking the thing out?
    N.B. because this is used for multi head attention typical inputs
    q, k, v have shape (batch_size, num_heads, seq_len, size)
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    The mask is multiplied with -1e9 (close to negative infinity). This is done 
    because the mask is summed with the scaled matrix multiplication of Q 
    and K and is applied immediately before a softmax. The goal is to zero
    out these cells, and large negative inputs to softmax are near zero 
    in the output.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """
    # does einsum require that you know the number of the dimensions, or does it allow for multiple batch dims?
    # if former:
    # unnorm_weights = tf.einsum('bhjk,bhik->bhij', k, q) / scale  # [B,m,n]

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    """
    Based on tf transformer tutorial:
      https://www.tensorflow.org/tutorials/text/transformer#encoder_layer

    N.B. that my original implementation allowed key dim to differ from value dim...
      I think this was to initialise the key weights differently from the 
      value weights, in case the keys and values were of different dimension.
      This was based on the initialisation used in the NeuralProcess implementation
      It might be worth comparing the implicit normalisation performed here
      (automatically by the dense layers) from the explicit normalisation performed in
      the neural process implementation...in fact I think the sqrt multiplication of
      the embeddings in the tf transformer tutorial might be connected to this.
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        """
        mask is a b, l node mask (see embedders.create_node_mask)
        `Mask out any nodes (cells/assays) which do not have any observed features (tracks)
        from the attention calculation: i.e. prevent them from being included in the normalisation.`
        """
        if mask is not None:
            tf.debugging.assert_rank(mask, 2)
        batch_size = tf.shape(q)[0]
        seq_len_q = tf.shape(q)[1]
        
        q = self.wq(q)  # (batch_size, q_len, d_model)
        k = self.wk(k)  # (batch_size, k_len, d_model)
        v = self.wv(v)  # (batch_size, k_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        # print("q shape", q.shape, "k shape", k.shape, "v shape", v.shape, "mask shape", mask.shape)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask[:, tf.newaxis, tf.newaxis, :])  # we need mask to be broadcastable to b, h, l, l
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        tf.debugging.assert_shapes([
            (q, ("B", self.num_heads, "L", self.depth)),
            (scaled_attention, ("B", "L", self.num_heads, self.depth))
        ])
        # expected_out_shape = (batch_size, seq_len_q, self.num_heads, self.depth)
        # assert scaled_attention.shape == expected_out_shape,\
        #     f"{scaled_attention.shape} != {expected_out_shape}"

        concat_attention = tf.reshape(scaled_attention, 
                                     (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return [output, attention_weights]

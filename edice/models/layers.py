import tensorflow as tf
import collections.abc
from tensorflow.keras import initializers

from edice.models.attention import ScaledDPSelfAttention, MultiHeadAttention


def batched_gather(params, indices, axis=None):
    """
    if you have (batch_size, params_dims) params
    and a different set of indices for each params item (i.e. batch_size, inds indices)
    then use this
    """
    return tf.gather(params, indices, batch_dims=1, axis=axis)


def output_mlp(d_output, d_hidden, n_hidden_layers=1, activation="relu",
               output_activation=None, dropout=0.):
    layers = []
    for i in range(n_hidden_layers):
        layers.append(tf.keras.layers.Dense(d_hidden, activation=activation))
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(d_output, activation=output_activation))
    return tf.keras.Sequential(layers)


class InputExpander(tf.keras.layers.Layer):

    """
    Receives as input a vector of inputs as well as corresponding cell, assay ids
    Returns a batch_size x n_nodes x n_feats tensor 
    (if input is obs_vec, cell_ids, assay_ids, and n_nodes is n_cells, n_feats is n_assays,
        then output is batch_size, n_cells, n_assays
     if input is obs_vec, assay_ids, cell_ids, and n_nodes is n_assays, n_feats is n_cells,
        then output is batch_size, n_assays, n_cells

    Output tensor has zeros in unobserved positions
    and has observed values filled in observed positions.

    cell_expander = InputExpander(n_cells, n_assays)
    cell_nodes = cell_expander(y_obs, cell_ids, assay_ids)
    """

    def __init__(self, n_nodes, n_feats, **kwargs):
        super().__init__(**kwargs)
        self.n_nodes = n_nodes
        self.n_feats = n_feats

    def call(self, inputs):
        obs_vec, node_ids, feat_ids = inputs # (batch_size, d)
        node_ids = tf.cast(node_ids, tf.int32)
        feat_ids = tf.cast(feat_ids, tf.int32)
        # can I potentially scatter_nd directly into a (batch_size, n_cells, n_assays) tensor
        batch_size = tf.shape(obs_vec)[0]
        batch_ids = tf.tile(tf.expand_dims(tf.range(tf.shape(obs_vec)[0]), axis=-1),
                           (1, tf.shape(obs_vec)[1])) # batch_size, d

        # create a single list containing tuples indexing into (b, n_nodes, n_feats) tensor
        # and also create a corresponding list of the values to be scattered into that tensor
        batch_ids = tf.reshape(batch_ids, [-1,1])
        node_ids = tf.reshape(node_ids, [-1,1])
        feat_ids = tf.reshape(feat_ids, [-1,1])
        obs_vec = tf.reshape(obs_vec, [-1,])  # just a 1D list. Using a non-tuple -1 threw error..
        
        observed_indices = tf.concat([batch_ids, node_ids, feat_ids], -1)
        # print(observed_indices.numpy())
        shape = [batch_size, self.n_nodes, self.n_feats]

        # identical to tensor_scatter_add(tf.zeros(shape, values.dtype), indices, values)
        obs_mat = tf.scatter_nd(observed_indices, obs_vec, shape)
        return obs_mat


class FlatInputMasker(tf.keras.layers.Layer):

    def call(self, inputs, training=None):
        obs_vec, mask = inputs[0], tf.cast(inputs[1], tf.float32)  # (batch_size, n_obs)
        obs_counts = tf.math.reduce_sum(mask, axis=-1, keepdims=True)  # (batch_size, 1)
        keep_prob = obs_counts / tf.cast(tf.shape(obs_vec)[1], tf.float32)
        scale = 1 / keep_prob
        if training:
            return obs_vec * scale * mask
        else:
            return obs_vec


class NodeInputMasker(tf.keras.layers.Layer):

    """
    Receives as input a tensor (batch_size, n_nodes, node_dim) inputs as well as a binary mask
    The mask will be 1 for all tensor entries which are observed and which
    are to be used as predictors
    Returns a masked input tensor, which is additionally scaled,
    (with the inputs for each 'node' divided by the number of observations for that node)
    together with counts of the number of observations in each node
    """

    def call(self, inputs):
        obs_mat, mask = inputs[0], inputs[1]
        obs_counts = tf.math.reduce_sum(mask, axis=-1, keepdims=True) # (batch_size, n_nodes, 1)
        obs_mat *= mask
        obs_mat = tf.math.divide_no_nan(tf.cast(obs_mat, tf.float32),
                                        tf.cast(obs_counts, tf.float32))  # 0/0 = 0 c.f. SO 45466142
        return obs_mat, obs_counts


class NodeAverager(tf.keras.layers.Layer):

    def __init__(self, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.keepdims = keepdims

    """
    Receives a (batch_size, n_nodes, node_dim) inputs tensor as well as a binary mask
    Returns a tensor of averages = sum(inputs) / sum(mask) for each node, using as denominator
    the number of observed dimensions rather than the number of overall dimensions
    """

    def call(self, inputs):
        obs_mat, mask = inputs[0], inputs[1]
        obs_counts = tf.math.reduce_sum(mask, axis=-1, keepdims=self.keepdims)
        node_totals = tf.math.reduce_sum(obs_mat, axis=-1, keepdims=self.keepdims)
        return tf.math.divide_no_nan(node_totals, tf.cast(obs_counts, tf.float32))


class TargetEmbeddingRetriever(tf.keras.layers.Layer):
    
    """
    given a (batch_size, n_targets) tensor of target item ids
    extract the corresponding slices from the (batch_size, n_items, embed_dim)
    embedding tensor
    """
    
    def call(self, inputs):
        all_embeddings, target_ids = inputs[0], inputs[1]
        return batched_gather(all_embeddings, target_ids, axis=1)


# Based on tf tutorial: https://www.tensorflow.org/tutorials/text/transformer#encoder_layer
class PreLNTransformerBlock(tf.keras.layers.Layer):

    """
    https://openreview.net/forum?id=B1x8anVFPr
    https://github.com/tensorflow/models/blob/09c5ae2fc056c620b83d51a9c5deece9a7e8fa49/official/nlp/transformer/transformer.py#L424
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1,
                 ffn_dropout=0.):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = output_mlp(d_model, dff, activation="relu",
                              dropout=ffn_dropout)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        xprime = self.layernorm1(x)
        attn_output, self.attn_weights = self.mha(xprime, xprime, xprime, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output

        ffn_output = self.ffn(self.layernorm2(out1))
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output
        return out2


class PostLNTransformerBlock(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1,
                 ffn_dropout=0.):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = output_mlp(d_model, dff, activation="relu",
                              dropout=ffn_dropout)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output, self.attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class NoLNTransformerBlock(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1,
                 ffn_dropout=0.):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = output_mlp(d_model, dff, activation="relu",
                              dropout=ffn_dropout)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output, self.attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x + attn_output  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output

        return out2


class CrossTransformerBlock(tf.keras.layers.Layer):

    """
    Whereas the standard transformer performs self-attention
    to transform a sequence of length L to a sequence of
    hidden representations of the same length,
    this transformer performs "cross-attention" (c.f. ANPs)
    (effectively filtered self-attention, only computing self-attention
        for a subset of the full set of nodes,
        with ids specified by the second arg (query_ids))

    N.B. if transformer block had same signature as MHA
    (i.e. taking x_v,x_k,x_q inputs, then this would be redundant)
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1,
                 ffn_dropout=0.):
        super().__init__()

        self.query_retriever = TargetEmbeddingRetriever()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = output_mlp(d_model, dff, activation="relu",
                              dropout=ffn_dropout)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, query_ids, training=None, mask=None):
        x_q = self.query_retriever([x, query_ids])  # (batch_size, n_queries, dim)
        attn_output, self.attn_weights = self.mha(x, x, x_q, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x_q + attn_output  # (batch_size, n_queries, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, n_queries, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output

        return out2


class NoAttnCrossTransformerBlock(tf.keras.layers.Layer):

    """
    No attention
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1,
                 ffn_dropout=0.):
        super().__init__()
        assert num_heads == 0
        self.query_retriever = TargetEmbeddingRetriever()
        self.ffn = output_mlp(d_model, dff, activation="relu",
                              dropout=ffn_dropout)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, query_ids, training=None, mask=None):
        x_q = self.query_retriever([x, query_ids])  # (batch_size, n_queries, dim)
        ffn_output = self.ffn(x_q)  # (batch_size, n_queries, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = x_q + ffn_output
        return out2


class SingleAttnCrossTransformerBlock(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1,
                 ffn_dropout=0., project_qkv=True):
        super().__init__()
        assert num_heads == 1
        self.query_retriever = TargetEmbeddingRetriever()
        self.self_attention = ScaledDPSelfAttention(d_model, project_qkv=project_qkv)
        self.ffn = output_mlp(d_model, dff, activation="relu",
                              dropout=ffn_dropout)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, query_ids, training=None, mask=None):
        x_q = self.query_retriever([x, query_ids])  # (batch_size, n_queries, dim)
        attn_output, self.attn_weights = self.self_attention(x, x, x_q, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = x_q + attn_output  # (batch_size, n_queries, d_model)

        ffn_output = self.ffn(x_q)  # (batch_size, n_queries, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = x_q + ffn_output
        return out2


class CrossSelfAttentionLayer(tf.keras.layers.Layer):

    """
    For purposes of ablation - removes all transformer bells and whistles
    other than residual connection, which is controllable
    """

    def __init__(self, d_model, project_qkv=True,
                 residual=True, rate=0.1):
        super().__init__()
        self.residual = residual
        self.query_retriever = TargetEmbeddingRetriever()
        self.self_attention = ScaledDPSelfAttention(d_model, project_qkv=project_qkv)
        self.dropout1 = tf.keras.layers.Dropout(rate)

    def call(self, x, query_ids, training=None, mask=None):
        x_q = self.query_retriever([x, query_ids])  # (batch_size, n_queries, dim)
        attn_output, self.attn_weights = self.self_attention(x, x, x_q, mask)
        attn_output = self.dropout1(attn_output, training=training)
        return x_q + attn_output if self.residual else attn_output

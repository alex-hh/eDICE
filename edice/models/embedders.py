import tensorflow as tf
from edice.models.layers import InputExpander, NodeInputMasker, NoLNTransformerBlock, CrossTransformerBlock


def create_node_mask(node_counts):
    """
    Mask out any nodes (cells/assays) which do not have any observed features (tracks)
    from the attention calculation: i.e. prevent them from being included in the normalisation.

    Mask computed from an input node_counts tensor (b, n_nodes) or (b, n_nodes, 1)
    Implementation is based on the tf transformer example:
    https://www.tensorflow.org/tutorials/text/transformer#masking

    How masking works:
    The mask is added to the ij attention weights tensor
    which is of size (batch_size, num_heads, num_queries, num_keys)
    the (b, n, i, j) element of this tensor is the softmax weight aij
    between the ith query and the jth key:

    aij = exp(qi . kj) / sum_m exp (qi . km) 

    Each query's weights are normalised to one over the keys
    Therefore, the important thing in the masking is to ensure
    that any keys that we dont want include do not contribute
    to the normalisation factor sum_m exp (qi . km)
    (any queries we don't want to contribute should just be
        excluded from the list of queries either at this
        point or later on they can just be ignored. In our
        case such queries may still be relevant - we might
        be predicting signal based on global embedding's
        similarity to other contextual cell / assay embeddings)
    To achieve the masking we write zij = qi . kj
    Then aij = exp(zij) / sum_m exp (zim) = softmax(zij)
    For any unobserved key ku we then set ziu = zij + 1e-9
    Then exp(ziu) = 0 so u does not contribute to the normalisation
    and aiu = 0.

    The idea is therefore to build a mask which is 1 for missing
    keys and 0 otherwise. We need to ensure that it is broadcastable
    to the shape (batch_size, num_heads, num_queries, num_keys) of zij
    """
    if tf.rank(node_counts) == 3:
        # last dim is a 1 d count (i.e. keepdims has been used with reduce sum)
        node_counts = tf.squeeze(node_counts, -1)
    missing_node_mask = tf.cast(tf.math.equal(node_counts, 0),
                                tf.float32)  # (b, n_nodes) [n_nodes = n_keys here]
    # missing_node_mask = missing_node_mask[:, tf.newaxis, tf.newaxis, :]  # (b, 1, 1, n_nodes)
    return missing_node_mask


class SignalEmbedder(tf.keras.layers.Layer):

    """
    Receives as input a vector of inputs, with corresponding track ids.
    Returns either a batch_size, n_cells, embed_dim or a batch_size, n_assays, embed_dim
    tensor of 'embedded' observations, formed by transforming the input signal via a 
    hidden layer, and adding global embeddings.
    """

    def __init__(self,
                 n_nodes,
                 n_feats,
                 embed_dim=32,
                 add_global_embedding=True,
                 embeddings_initializer='uniform',
                 dropout=0.,
                 **kwargs):
        super().__init__(**kwargs)
        self.expander = InputExpander(n_nodes, n_feats)
        self.mask_scaler = NodeInputMasker()
        self.nodewise_hidden = tf.keras.layers.Dense(embed_dim, activation='relu')
        self.add_global_embedding = add_global_embedding
        self.embeddings_initializer = embeddings_initializer
        self.embed_dim = embed_dim
        self.n_nodes = n_nodes
        self.n_feats = n_feats
        self.dropout = tf.keras.layers.Dropout(dropout)
        print(f"Instantiated embedder interpreting signal as a graph of {self.n_nodes} nodes and {self.n_feats} feats")
        
    def build(self, input_shape):
        if self.add_global_embedding:
            self.global_embeddings = self.add_weight(
                shape=(self.n_nodes, self.embed_dim),
                initializer=self.embeddings_initializer,
                name='global_embeddings')

    def call(self, inputs, training=None):
        obs_vec, node_ids, feat_ids = inputs
        obs_mat = self.expander([obs_vec, node_ids, feat_ids])  # (b, n_nodes, n_feats)
        # build binary mask: 1 for observed entries in obs_mat, 0 for unobserved
        mask_mat = self.expander([tf.ones_like(obs_vec), node_ids, feat_ids])  # (b, n_nodes, n_feats)
        # scales each cell|assay's inputs by number observed tracks in that cell|assay
        obs_masked, obs_counts = self.mask_scaler([obs_mat, mask_mat])
        missing_node_mask = create_node_mask(obs_counts)
        embedded = self.nodewise_hidden(obs_masked)
        if self.add_global_embedding:
            # embedded += self.global_embedding(tf.tile(tf.expand_dims(tf.range(self.expander.n_nodes),0), (b, 1))
            # obviously global embeddings dont have a batch dim but I guess broadcasting takes care of this?
            embedded += self.global_embeddings
        embedded = self.dropout(embedded, training=training)
        return embedded, missing_node_mask


class CrossContextualSignalEmbedder(tf.keras.layers.Layer):

    """
    Uses cross attention rather than self attention for final attention layer.
    """

    def __init__(self,
                 n_nodes,
                 n_feats,
                 embed_dim=32,
                 n_attn_layers=1,
                 n_attn_heads=4,
                 add_global_embedding=True,
                 intermediate_fc_dim=128,
                 transformer_dropout=0.1,
                 intermediate_fc_dropout=0.,
                 embedding_dropout=0.,
                 transformer_kwargs=None,
                 **kwargs):
        super().__init__(**kwargs)
        transformer_kwargs = transformer_kwargs or {} # add interspersed_fc_dim 128
        self.signal_embedder = SignalEmbedder(n_nodes, n_feats,
                                              embed_dim=embed_dim,
                                              dropout=embedding_dropout,
                                              add_global_embedding=add_global_embedding)
        self.n_attn_layers = n_attn_layers

        self_transformer = NoLNTransformerBlock
        assert n_attn_layers > 0, "Implementation assumes n_attn_layers >= 0"

        for i in range(n_attn_layers-1):
            setattr(self, f'transformer_{i}',
                    self_transformer(embed_dim, n_attn_heads,
                                     intermediate_fc_dim,
                                     rate=transformer_dropout,
                                     ffn_dropout=intermediate_fc_dropout,
                                     **transformer_kwargs))

        assert n_attn_heads > 0:

        setattr(self, f'transformer_{n_attn_layers-1}',
                CrossTransformerBlock(
                    embed_dim, n_attn_heads, intermediate_fc_dim,
                    rate=transformer_dropout,
                    ffn_dropout=intermediate_fc_dropout,
                    **transformer_kwargs
                )
            )

    def call(self, inputs, training=None):
        obs_vec, node_ids, feat_ids, query_ids = inputs
        embedded, node_mask = self.signal_embedder([obs_vec, node_ids, feat_ids],
                                                   training=training)

        for i in range(self.n_attn_layers-1):
            embedded = getattr(self, f'transformer_{i}')(
                embedded, training=training, mask=node_mask)
        embedded = getattr(self, f'transformer_{self.n_attn_layers-1}')(
            embedded, query_ids, training=training, mask=node_mask)

        return embedded

import tensorflow as tf

from edice.models.base import BaseModel
from edice.models.embedders import CrossContextualSignalEmbedder as CrossSignalEmbedder
from edice.models.layers import InputExpander, NodeAverager, FlatInputMasker,\
                          TargetEmbeddingRetriever, output_mlp


class AssayAverager(BaseModel):

    def __init__(self, n_assays, n_cells, **kwargs):
        super().__init__(**kwargs)
        self.expander = InputExpander(n_assays, n_cells)
        self.averager = NodeAverager(keepdims=True)
        self.prediction_retriever = TargetEmbeddingRetriever()

    def call(self, inputs):
        supports, support_cell_ids, support_assay_ids = inputs[0], inputs[1], inputs[2]
        target_cell_ids, target_assay_ids = inputs[3], inputs[4]
        obs_mat = self.expander([supports, support_assay_ids, support_cell_ids])  # (b, n_assays, n_cells)
        # build binary mask: 1 for observed entries in obs_mat, 0 for unobserved
        mask_mat = self.expander([tf.ones_like(supports), support_assay_ids, support_cell_ids])
        assay_averages = self.averager([obs_mat, mask_mat]) # (b, n_assays, 1)
        preds = self.prediction_retriever([assay_averages, target_assay_ids])
        return tf.squeeze(preds, axis=-1)


class CellAssayCrossFactoriser(BaseModel):

    def __init__(self,
                 n_cells,
                 n_assays,
                 transformation="arcsinh",
                 embed_dim=32,
                 n_attn_layers=1,
                 n_attn_heads=4,
                 intermediate_fc_dim=128,
                 layer_norm_type=None,
                 decoder_layers=2,
                 decoder_hidden=2048,
                 decoder_dropout=0.3,
                 transformer_dropout=0.1,
                 intermediate_fc_dropout=0.,
                 embedding_dropout=0.,
                 single_head=False,
                 single_head_residual=True,
                 **kwargs):
        super().__init__(transformation=transformation, **kwargs)
        self.n_cells = n_cells
        self.n_assays = n_assays
        self.n_attn_heads = n_attn_heads
        print("Embed dim", embed_dim)
        signal_embedder = CrossSignalEmbedder
        self.cell_embedder = signal_embedder(self.n_cells, self.n_assays,
                                             embed_dim=embed_dim,
                                             n_attn_layers=n_attn_layers,
                                             n_attn_heads=n_attn_heads,
                                             intermediate_fc_dim=intermediate_fc_dim,
                                             layer_norm_type=layer_norm_type,
                                             transformer_dropout=transformer_dropout,
                                             intermediate_fc_dropout=intermediate_fc_dropout,
                                             embedding_dropout=embedding_dropout,
                                             single_head=single_head,
                                             single_head_residual=single_head_residual,
                                             name="contextual_cell_embedder")
        self.assay_embedder = signal_embedder(self.n_assays, self.n_cells,
                                              embed_dim=embed_dim,
                                              n_attn_layers=n_attn_layers,
                                              n_attn_heads=n_attn_heads,
                                              intermediate_fc_dim=intermediate_fc_dim,
                                              layer_norm_type=layer_norm_type,
                                              transformer_dropout=transformer_dropout,
                                              intermediate_fc_dropout=intermediate_fc_dropout,
                                              embedding_dropout=embedding_dropout,
                                              single_head=single_head,
                                              single_head_residual=single_head_residual,
                                              name="contextual_assay_embedder")
        self.output_mlp = output_mlp(1, decoder_hidden, n_hidden_layers=decoder_layers,
                                     activation="relu", dropout=decoder_dropout)

    def call(self, inputs, training=None):
        supports, support_cell_ids, support_assay_ids = inputs[0], inputs[1], inputs[2]
        target_cell_ids, target_assay_ids = inputs[3], inputs[4]
        target_cell_embeddings = self.cell_embedder(
            [supports, support_cell_ids, support_assay_ids, target_cell_ids],
            training=training)
        target_assay_embeddings = self.assay_embedder(
            [supports, support_assay_ids, support_cell_ids, target_assay_ids],
            training=training)
        mlp_inputs = tf.concat([target_cell_embeddings, target_assay_embeddings], -1)
        return tf.squeeze(self.output_mlp(mlp_inputs), -1)


class ChallengeModel(BaseModel):

    """
    N.B. Challenge model requires a different data loader - because it can't 
    use variable sized slices defined by support cell/assay ids as inputs, 
    it needs a fixed-size input. This is handled by passing fixed_inputs=True
    to BaseModel init as bleow
    """
    
    def __init__(self, n_cells, n_assays, 
                 encoder_dims=[128,128], decoder_hidden=2048,
                 embed_dim=32, decoder_dropout=0.4, decoder_layers=2, **kwargs):
        super().__init__(fixed_inputs=True, **kwargs)
        self.cell_embeddings = tf.keras.layers.Embedding(n_cells, embed_dim)
        self.assay_embeddings = tf.keras.layers.Embedding(n_assays, embed_dim)
        self.bin_masker = FlatInputMasker()
        self.bin_encoder = tf.keras.Sequential([
                    tf.keras.layers.Dense(d, activation='relu')
                    for d in encoder_dims])
        self.output_mlp = output_mlp(
            1, decoder_hidden, n_hidden_layers=decoder_layers,
            dropout=decoder_dropout)

    def call(self, inputs, training=False):
        supports, mask, target_cell_ids, target_assay_ids = inputs
        masked_supports = self.bin_masker([supports, mask])
        encoded_bin = self.bin_encoder(masked_supports)
        target_cell_embeddings = self.cell_embeddings(target_cell_ids)
        target_assay_embeddings = self.assay_embeddings(target_assay_ids)
        encoded_bin_tiled = tf.tile(tf.expand_dims(encoded_bin, 1),
                                    (1, tf.shape(target_cell_ids)[-1], 1))
        # (b, n_targets, total_embed_dim)
        target_embeddings = tf.concat([target_cell_embeddings,
                                       target_assay_embeddings,
                                       encoded_bin_tiled], -1)
        # print("decoder inputs", target_embeddings)
        # how can training arg be passed to sublayers?
        return tf.squeeze(self.output_mlp(target_embeddings), -1)

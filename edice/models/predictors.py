import tensorflow as tf

from edice.models.base import BaseModel
from edice.models.embedders import CrossContextualSignalEmbedder as CrossSignalEmbedder
from edice.models.layers import output_mlp


class CellAssayCrossFactoriser(BaseModel):

    def __init__(self,
                 n_cells,
                 n_assays,
                 transformation="arcsinh",
                 embed_dim=32,
                 n_attn_layers=1,
                 n_attn_heads=4,
                 intermediate_fc_dim=128,
                 decoder_layers=2,
                 decoder_hidden=2048,
                 decoder_dropout=0.3,
                 transformer_dropout=0.1,
                 intermediate_fc_dropout=0.,
                 embedding_dropout=0.,
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
                                             transformer_dropout=transformer_dropout,
                                             intermediate_fc_dropout=intermediate_fc_dropout,
                                             embedding_dropout=embedding_dropout,
                                             name="contextual_cell_embedder")
        self.assay_embedder = signal_embedder(self.n_assays, self.n_cells,
                                              embed_dim=embed_dim,
                                              n_attn_layers=n_attn_layers,
                                              n_attn_heads=n_attn_heads,
                                              intermediate_fc_dim=intermediate_fc_dim,
                                              transformer_dropout=transformer_dropout,
                                              intermediate_fc_dropout=intermediate_fc_dropout,
                                              embedding_dropout=embedding_dropout,
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

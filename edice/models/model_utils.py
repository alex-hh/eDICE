from tensorflow.keras.optimizers import Adam

from edice.models import predictors


def get_factoriser_model(config, n_cells, n_assays):
    model_config = dict(n_cells=n_cells,
                        n_assays=n_assays,
                        embed_dim=config.embed_dim,
                        decoder_layers=config.decoder_layers,
                        decoder_hidden=config.decoder_hidden,
                        decoder_dropout=config.decoder_dropout,
                        transformation=config.transformation,
                        n_attn_layers=config.n_attn_layers,
                        n_attn_heads=config.n_attn_heads,
                        intermediate_fc_dim=config.intermediate_fc_dim,
                        transformer_dropout=getattr(config, "transformer_dropout", 0.1),
                        intermediate_fc_dropout=getattr(config, "intermediate_fc_dropout", 0.),
                        embedding_dropout=getattr(config, "embedding_dropout", 0.))
    assert getattr(config, "cellagg", None) is None and config.n_attn_layers > 0
    model_class = "CellAssayCrossFactoriser"
    print("Loading model class", model_class)
    model = getattr(predictors, model_class)(**model_config)
    return model


def load_model(n_cells, n_assays, config, compile_model=True):
    model = get_factoriser_model(config, n_cells, n_assays)
    
    if compile_model:
        model.compile(loss="mse",
                      optimizer=Adam(learning_rate=config.lr),
                      run_eagerly=config.test_run)
    return model

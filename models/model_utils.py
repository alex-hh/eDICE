from tensorflow.keras.optimizers import Adam

from models import predictors


def get_challenge_model(config, n_cells, n_assays):
    model_config = dict(n_cells=n_cells,
                        n_assays=n_assays,
                        embed_dim=config.embed_dim,
                        decoder_layers=config.decoder_layers,
                        decoder_hidden=config.decoder_hidden,
                        decoder_dropout=config.decoder_dropout,
                        transformation=config.transformation)
    model = predictors.ChallengeModel(**model_config)
    return model


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
                        embedding_dropout=getattr(config, "embedding_dropout", 0.),
                        single_head=getattr(config, "single_head", False),
                        single_head_residual=getattr(config, "single_head_residual", True),
                        layer_norm_type=config.layer_norm_type)
    assert getattr(config, "cellagg", None) is None and config.n_attn_layers > 0:
    model_class = "CellAssayCrossFactoriser"
    print("Loading model class", model_class)
    model = getattr(predictors, model_class)(**model_config)
    return model


def load_model(n_cells, n_assays, config, compile_model=True):
    """
    might need to handle different models slightly differently
    e.g. probabilistic models might have some extra args or sthg?
    how to handle this: if models have mostly overlapping arguments,
    but a few different ones? Maybe have model constructors
    which explicitly instantiate the model with the relevant args...
    """

    if getattr(config, "assayavg", False):
        print("Using assay average baseline")
        model = predictors.AssayAverager(n_assays, n_cells)
        if compile_model:
            model.compile(loss="mse", optimizer="adam",
                          run_eagerly=config.test_run)
        return model

    else:
        if getattr(config, "challenge", False):
            print("Using challenge model")
            model = get_challenge_model(config, n_cells, n_assays)
        else:
            # config common to all model classes
            model = get_factoriser_model(config, n_cells, n_assays)
            
        if compile_model:
            model.compile(loss=config.loss if config.output_dist is None else None,
                          optimizer=Adam(learning_rate=config.lr),
                          run_eagerly=config.test_run)
        return model

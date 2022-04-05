from multiprocessing.sharedctypes import Value
import sys
import numpy as np
import pandas as pd
import h5py
import itertools
import tensorflow as tf
from tqdm import tqdm
from argparse import ArgumentParser
import os
from os.path import join
from argparse import Namespace
from tensorflow import keras
import tensorflow.keras.backend as K

from data_loaders.dataset_config import load_dataset
from utils.train_utils import load_saved_config, get_output_dir, ConfigSaver
from models.model_utils import load_model
from data_loaders.data_generators import TrainInMemGenerator, ValInMemGenerator
from utils.callbacks import Checkpoint, RunningMetricPrinter



individuals = ["male_37", "male_54", "female_53", "female_51"]

selected_common_tracks = ['E065-H3K27ac',
 'E065-H3K36me3',
 'E065-H3K4me1',
 'E065-H3K4me3',
 'E065-H3K9me3',
 'E079-H3K27ac',
 'E079-H3K36me3',
 'E079-H3K4me1',
 'E079-H3K4me3',
 'E079-H3K9me3',
 'E094-H3K27ac',
 'E094-H3K36me3',
 'E094-H3K4me1',
 'E094-H3K4me3',
 'E094-H3K9me3',
 'E096-H3K27ac',
 'E096-H3K36me3',
 'E096-H3K4me1',
 'E096-H3K4me3',
 'E096-H3K9me3',
 'E106-H3K27ac',
 'E106-H3K36me3',
 'E106-H3K4me1',
 'E106-H3K4me3',
 'E106-H3K9me3',
 'E113-H3K27ac',
 'E113-H3K4me1',
 'E113-H3K4me3',
 'E113-H3K9me3']


TRAINING_SETUPS = list(itertools.product(individuals, selected_common_tracks))
entex_assay2id = {"H3K36me3":0, "H3K4me1":1, "H3K4me3":2, "H3K9me3":3, "H3K27ac":4}
entex_cell2id = {"E065": 0, "E079": 1, "E094":2, "E096":3, "E106": 4, "E113": 5}



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--training_setup", type=int, default=0)
    parser.add_argument("--data_folder", type=str,
                        default="ENTEx/processed_data")
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--exp_name", default="test")
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--transfer_from", type=str, default="ENTEx")
    parser.add_argument("--blacklist_path", type=str, default="localdata/hg19-blacklist.v2.bed")
    args = parser.parse_args()

    # Load the dataset class for the Roadmap dataset to have access to precomputed gap masks
    dataset = load_dataset("PredictdChr21")
    mask = dataset._get_bin_mask(exclude_gaps=True)


    # Filter the Blacklisted regions in chromosome 21
    blacklist = pd.read_csv(args.blacklist_path, sep="\t", header=None)
    blacklist.columns = ["Chromosome", "Start (bp)", "End (bp)", "Description"]
    blacklist = blacklist[blacklist["Chromosome"] == "chr21"]
    blacklist_mask = np.ones_like(mask)
    for i, row in blacklist.iterrows():
        st_bin = row["Start (bp)"]//25
        end_bin = row["End (bp)"]//25
        blacklist_mask[st_bin:end_bin] = 0
    blacklist_mask_no_gaps = blacklist_mask[mask].astype(bool)


    individual_label, target_track = TRAINING_SETUPS[args.training_setup]
    individual_idx = individuals.index(individual_label)
    print("\n\nTraining Setup: ", individual_label, target_track)

    # Create output directory where predicted tracks will be saved
    experiment_results_dir = "ENTEx/results_{}".format(args.exp_name)
    if os.path.exists(join(experiment_results_dir, "{}_{}_imputed.npy".format(individual_label, target_track))):
        print("Track already processed, quitting\n\n")
        sys.exit()
    if not os.path.exists(experiment_results_dir):
        try:
            os.makedirs(experiment_results_dir)
        except FileExistsError:
            pass

    # Select the tracks to use as support for training and prediction
    support_tracks = sorted([t for t in selected_common_tracks if t!=target_track])

    ntargets = max(1, min(10, int(0.2*len(support_tracks))))

    # Select configurations changes based on which dataset we transfer from
    if args.transfer_from =="ENTEx":
        n_cells = 6  # roadmap default
        n_assays = 5  # roadmap default
        cell2id = entex_cell2id
        assay2id = entex_assay2id
    elif args.transfer_from=="Roadmap":
        n_cells = 127  # roadmap default
        n_assays = 24  # roadmap default
        cell2id = dataset.cell2id
        assay2id = dataset.assay2id
    else:
        raise ValueError("Invalid dataset for transfer")

    # Modify the configuration to reflect the finetuned model
    model_config = load_saved_config(args.model_config, "gv")
    cfg = Namespace(**model_config)
    cfg.lr = args.learning_rate
    cfg.experiment_name = individual_label + "_" + args.exp_name + "_" + target_track
    cfg.experiment_group = "ENTEx_finetune"
    cfg.epochs = args.n_epochs
    cfg.ntargets = ntargets

    # Instantiate the model
    model = load_model(n_cells, n_assays, cfg)

    # SANITY CHECK FOR LOADED MODEL
    nn = 32
    supports = np.zeros((nn, 20))
    cell_ids, assay_ids = np.zeros((nn, 20)).astype(
        int), np.zeros((nn, 20)).astype(int)
    tcell_ids, tassay_ids = np.zeros((nn, 10)).astype(
        int), np.zeros((nn, 10)).astype(int)
    print(model([supports, cell_ids, assay_ids,
                 tcell_ids, tassay_ids]).shape)

    # Load weights from checkpoint
    model.load_weights(args.checkpoint_path)

    K.set_value(model.optimizer.learning_rate, args.learning_rate)
    print("Setting learning rate to: ", K.eval(model.optimizer.lr))

    ## Freeze layers
    model.assay_embedder.signal_embedder.global_embeddings = tf.Variable(model.assay_embedder.signal_embedder.global_embeddings.value(), trainable=False)
    model.assay_embedder.transformer_0.trainable = False
    model.cell_embedder.signal_embedder.global_embeddings = tf.Variable(model.cell_embedder.signal_embedder.global_embeddings.value(), trainable=False)
    model.cell_embedder.transformer_0.trainable = False


    # Reading the individual's data
    with h5py.File(join(args.data_folder, "{}_r.hdf5".format(individual_label)), "r") as f:
        print("Loading data...")
        tracks = f["tracks"].attrs["track_names"]
        idxs = [np.where(tracks==t)[0][0] for t in support_tracks]

        support_assays = [t.split("-")[1] for t in support_tracks]
        support_cells = [t.split("-")[0] for t in support_tracks]
        support_assay_ids = [assay2id[a] for a in support_assays]
        support_cell_ids = [cell2id[c] for c in support_cells]
        test_idxs = [i for i, t in enumerate(tracks) if t==target_track]
        data = f["tracks"][:, mask]
        test_data = data[test_idxs, :].T
        data = data[idxs, :].T
        print("Loaded tracks\n")

    # Select where the finetuned model and its configuration are saved
    output_dir = get_output_dir(cfg.experiment_name, cfg.experiment_group,
                                resume=False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    callbacks = [
        RunningMetricPrinter(log_freq=500),
        ConfigSaver(
            args, output_dir / "config.yaml",
            splits=None, format="yaml")
    ]


    # Tune the model
    print("Initializing training generator...")
    training_generator = TrainInMemGenerator(
        data, support_cell_ids, support_assay_ids, transform='arcsinh', n_targets=ntargets)
    print("Training generator available\n")
    model.fit(training_generator, callbacks=callbacks, epochs=args.n_epochs)

    del training_generator

    # Predict the target track
    test_assays = [target_track.split("-")[1]]
    test_cells = [target_track.split("-")[0]]
    test_assay_ids = [assay2id[a] for a in test_assays]
    test_cell_ids = [cell2id[c] for c in test_cells]

    print("Initializing test generator...")
    test_generator = ValInMemGenerator(
        data, support_cell_ids, support_assay_ids, test_data, test_cell_ids, test_assay_ids, transform='arcsinh')

    predicted_track = np.zeros_like(test_data.flatten())

    index = 0
    for i_batch in tqdm(range(len(test_generator))):
        batch, targets = test_generator[i_batch]
        batch_len = len(batch[0])
        predictions = model(batch, training=False).numpy().flatten()
        predicted_track[index:index+batch_len] = predictions
        index += batch_len
    if not os.path.exists(experiment_results_dir):
        os.makedirs(experiment_results_dir)
    np.save(join(experiment_results_dir, "{}_{}_imputed.npy".format(individual_label, target_track)), predicted_track)

    print("\nPrediction Complete!")

import sys
import numpy as np
import pandas as pd
import h5py
import itertools
import yaml
import tensorflow as tf
from tqdm import tqdm
from argparse import ArgumentParser
import os
from os.path import join
from argparse import Namespace
from tensorflow import keras
from tensorflow.keras import backend as K

from edice.utils.train_utils import get_output_dir, ConfigSaver
from edice.models.model_utils import load_model
from edice.data_loaders.data_generators import TissueLOO_TrainInMemGenerator, ValInMemGenerator
from sklearn.metrics import auc, precision_recall_curve


individuals = ["male_37", "male_54", "female_53", "female_51"]

selected_common_tracks = ['aorta-H3K27ac',
 'aorta-H3K36me3',
 'aorta-H3K4me1',
 'aorta-H3K4me3',
 'aorta-H3K9me3',
 'esophagus_muscularis_mucosa-H3K27ac',
 'esophagus_muscularis_mucosa-H3K36me3',
 'esophagus_muscularis_mucosa-H3K4me1',
 'esophagus_muscularis_mucosa-H3K4me3',
 'esophagus_muscularis_mucosa-H3K9me3',
 'stomach-H3K27ac',
 'stomach-H3K36me3',
 'stomach-H3K4me1',
 'stomach-H3K4me3',
 'stomach-H3K9me3',
 'upper_lobe_of_left_lung-H3K27ac',
 'upper_lobe_of_left_lung-H3K36me3',
 'upper_lobe_of_left_lung-H3K4me1',
 'upper_lobe_of_left_lung-H3K4me3',
 'upper_lobe_of_left_lung-H3K9me3',
 'sigmoid_colon-H3K27ac',
 'sigmoid_colon-H3K36me3',
 'sigmoid_colon-H3K4me1',
 'sigmoid_colon-H3K4me3',
 'sigmoid_colon-H3K9me3',
 'spleen-H3K27ac',
 'spleen-H3K4me1',
 'spleen-H3K4me3',
 'spleen-H3K9me3']

tissues = ['aorta',
 'esophagus_muscularis_mucosa',
 'stomach',
 'upper_lobe_of_left_lung',
 'sigmoid_colon',
 'spleen']
assays = ["H3K36me3", "H3K4me1", "H3K4me3", "H3K9me3", "H3K27ac"]



assay2id = {"H3K36me3":0, "H3K4me1":1, "H3K4me3":2, "H3K9me3":3, "H3K27ac":4}
cell2id = {'aorta': 0,
 'esophagus_muscularis_mucosa': 1,
 'stomach': 2,
 'upper_lobe_of_left_lung': 3,
 'sigmoid_colon': 4,
 'spleen': 5}

TRAINING_SETUPS = list(itertools.product(individuals, individuals, tissues))


def auprc(target, pred, **kwargs):
    precision, recall, thresholds = precision_recall_curve(target, pred)
    return auc(recall, precision)

def mse(y_true, y_pred):
    return np.square(y_true - y_pred).mean()


def gwcorr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]



def main(args):
    mask = np.load("sample_data/annotations/hg19_ch21_gaps_mask.npy").astype(bool) 
    n_cells = len(tissues)
    n_assays = len(assays)

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


    # Create output directory where predicted tracks will be saved
    experiment_results_dir = "outputs/results_{}/{}".format(args.experiment_group, args.experiment_name)
    if os.path.exists(join(experiment_results_dir, "trsf-{}_tgt-{}_tgttissue-{}_metrics.csv".format(
                                                    args.transfer_individual, args.target_individual, args.target_tissue))):
        print("Tissue already processed, quitting\n\n")
        sys.exit()
    if not os.path.exists(experiment_results_dir):
        try:
            os.makedirs(experiment_results_dir)
        except FileExistsError:
            pass

    if args.training_setup==0:
        config_dict = vars(args)
        with open(join(experiment_results_dir, "training_config.yml"), "w") as f:
            yaml.dump(config_dict, f)

    # Select where the finetuned model and its configuration are saved
    output_dir = get_output_dir(args.experiment_name+str(args.training_setup), args.experiment_group,
                                resume=False)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    callbacks = [
        # RunningMetricPrinter(log_freq=500),
        ConfigSaver(
            args, output_dir / "config.yaml",
            splits=None, format="yaml")
    ]



    # Instantiate the model
    print("Loading model", flush=True)
    model = load_model(n_cells, n_assays, args, compile_model=True)

    #######################################################################
    ## First round of training: full dataset for transfer individual

    # Select the tracks to use as support for training and prediction
    support_tracks = selected_common_tracks

    # Reading the transfer individual's data
    transfer_ind_fname = join(args.data_folder, "ENTEx_{}_chr21.hdf5".format(args.transfer_individual))
    if not os.path.exists(transfer_ind_fname):
        raise FileNotFoundError("Transfer individual data not found")
    with h5py.File(transfer_ind_fname, "r") as f:
        print("Loading data...")
        tracks = f["tracks"].attrs["track_names"]
        target_tracks = [t for t in support_tracks if args.target_tissue in t]
        transfer_support_tracks = [t for t in support_tracks if args.target_tissue not in t]
        idxs = [np.where(tracks==t)[0][0] for t in support_tracks]
        transfer_idxs = [np.where(tracks==t)[0][0] for t in transfer_support_tracks]
        target_idxs = [np.where(tracks==t)[0][0] for t in target_tracks]

        support_assays = [t.split("-")[1] for t in support_tracks]
        support_cells = [t.split("-")[0] for t in support_tracks]
        support_assay_ids = [assay2id[a] for a in support_assays]
        support_cell_ids = [cell2id[c] for c in support_cells]


        transfer_assays = [t.split("-")[1] for t in transfer_support_tracks]
        transfer_cells = [t.split("-")[0] for t in transfer_support_tracks]
        transfer_assay_ids = [assay2id[a] for a in transfer_assays]
        transfer_cell_ids = [cell2id[c] for c in transfer_cells]
        data = f["tracks"][:, mask]
        transfer_data = data[transfer_idxs, :].T
        target_data = data[target_idxs, :].T
        data = data[idxs, :].T
        print("Loaded tracks\n")

    print("Start training..")


    print("Initializing training generator...")
    training_generator = TissueLOO_TrainInMemGenerator(
        data, support_cell_ids, support_assay_ids, transform='arcsinh')
    
    print("Training generator available\n")
    model.fit(training_generator, callbacks=callbacks, epochs=args.epochs)

    del training_generator


    print("Initializing finetuning generator...")

    test_assays = [t.split("-")[1] for t in target_tracks]
    test_cells = [t.split("-")[0] for t in target_tracks]
    test_assay_ids = [assay2id[a] for a in test_assays]
    test_cell_ids = [cell2id[c] for c in test_cells]
    checkpoints_path = os.path.join(output_dir, "ft_checkpoints")

    K.set_value(model.optimizer.learning_rate, args.ft_learning_rate)
    ##############################################################################################
    # Predicting target individual tracks
    # Reading the transfer individual's data
    
    target_ind_fname = join(args.data_folder, "ENTEx_{}_chr21.hdf5".format(args.target_individual))
    if not os.path.exists(target_ind_fname):
        raise FileNotFoundError("Target individual data not found")
    
    with h5py.File(target_ind_fname, "r") as f:
        print("Loading target individual data...")
        tracks = f["tracks"].attrs["track_names"]
        target_tracks = [t for t in support_tracks if args.target_tissue in t]
        transfer_support_tracks = [t for t in support_tracks if args.target_tissue not in t]
        transfer_idxs = [np.where(tracks==t)[0][0] for t in transfer_support_tracks]
        target_idxs = [np.where(tracks==t)[0][0] for t in target_tracks]

        transfer_assays = [t.split("-")[1] for t in transfer_support_tracks]
        transfer_cells = [t.split("-")[0] for t in transfer_support_tracks]
        transfer_assay_ids = [assay2id[a] for a in transfer_assays]
        transfer_cell_ids = [cell2id[c] for c in transfer_cells]
        data = f["tracks"][:, mask]
        transfer_data = data[transfer_idxs, :].T
        target_data = data[target_idxs, :].T
        print("Loaded tracks\n")

    

    target_ft_generator = TissueLOO_TrainInMemGenerator(
        transfer_data, transfer_cell_ids, transfer_assay_ids, transform='arcsinh')
    print("Finetuning target generator available\n")
    ft_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss', mode='min'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, monitor='loss', mode='min', save_weights_only=True, save_best_only=True)
    ]
    model.fit(target_ft_generator, callbacks=ft_callbacks, epochs=args.ft_epochs)

    # Restoring best fine-tuned model
    model.load_weights(checkpoints_path)

    
    test_generator = ValInMemGenerator(
        transfer_data, transfer_cell_ids, transfer_assay_ids, target_data, test_cell_ids, test_assay_ids, transform='arcsinh')


    index = 0
    predicted_tracks = np.zeros_like(target_data)

    for i_batch in tqdm(range(len(test_generator))):
        batch, targets = test_generator[i_batch]
        batch_len = len(batch[0])
        predictions = model(batch, training=False).numpy()
        predicted_tracks[index:index+batch_len, :] = predictions
        index += batch_len
    if not os.path.exists(experiment_results_dir):
        os.makedirs(experiment_results_dir)
    np.save(join(experiment_results_dir, "trsf-{}_tgt-{}_tgttissue-{}_imputed.npy".format(
                                                    args.transfer_individual, args.target_individual, args.target_tissue)), predicted_tracks)



    print("Analysis complete!")



MODEL_DEFAULTS = dict(layer_norm_type=None,
                        decoder_layers=2,
                        decoder_hidden=512,
                        decoder_dropout=0.3,
                        total_bins=None)

if __name__ == "__main__":
    parser = ArgumentParser()
    
    
    parser.add_argument("--training_setup", type=int, default=5)
    parser.add_argument("--transfer_individual", type=str)
    parser.add_argument("--target_individual", type=str)
    parser.add_argument("--target_tissue", type=str)

    parser.add_argument("--data_folder", type=str,
                        default="data/ENTEx/processed_data")
    parser.add_argument("--ft_learning_rate", type=float, default=0.00003)
    parser.add_argument("--experiment_name", default="ENTEX_finetune")
    parser.add_argument("--experiment_group", default="LOO_tissue_transfer")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--ft_epochs", default=15, type=int)
    parser.add_argument("--blacklist_path", type=str, default="sample_data/annotations/hg19-blacklist.v2.bed")


    parser.add_argument('--train_splits', type=str, default=["train"], nargs="+")
    parser.add_argument('--challenge', action="store_true")
    parser.add_argument('--test_run', action="store_true")
    parser.add_argument('--seed', default=43, type=int)

    parser.add_argument('--transformation', type=str, choices=["arcsinh"], default="arcsinh")
    
    parser.add_argument('--n_attn_heads', type=int, default=4)
    parser.add_argument('--n_attn_layers', type=int, default=1)
    parser.add_argument('--single_head', action="store_true")
    parser.add_argument('--single_head_residual', action="store_true")
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--layer_norm', type=str, choices=["pre", "post"], default=None)
    parser.add_argument('--intermediate_fc_dim', type=int, default=128)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--embedding_dropout', type=float, default=0.)
    parser.add_argument('--intermediate_fc_dropout', type=float, default=0.)

    
    parser.add_argument('--checkpoint', action="store_true")

    parser.set_defaults(**MODEL_DEFAULTS)
    args = parser.parse_args()

    if args.training_setup is not None:
        ts = TRAINING_SETUPS[args.training_setup]
        args.transfer_individual, args.target_individual, args.target_tissue = ts


    main(args)
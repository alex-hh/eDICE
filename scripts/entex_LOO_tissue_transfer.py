import sys
import numpy as np
import pandas as pd
import h5py
import itertools
from torch import t
import yaml
import tensorflow as tf
from tqdm import tqdm
from argparse import ArgumentParser
import os
from os.path import join
from argparse import Namespace
from tensorflow import keras
from tensorflow.keras import backend as K
# import tensorflow.keras.backend as K

from edice.utils.train_utils import get_output_dir, ConfigSaver
from edice.models.model_utils import load_model
from edice.data_loaders.data_generators import TissueLOO_TrainInMemGenerator, ValInMemGenerator
from edice.utils.callbacks import Checkpoint, RunningMetricPrinter
from sklearn.metrics import auc, precision_recall_curve


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


assay2id = {"H3K36me3":0, "H3K4me1":1, "H3K4me3":2, "H3K9me3":3, "H3K27ac":4}
cell2id = {"E065": 0, "E079": 1, "E094":2, "E096":3, "E106": 4, "E113": 5}

ind_pairings ={"male_37": "male_54",
                "male_54": "male_37",
                "female_53": "female_51",
                "female_51": "female_53"}

tissues = ["E065", "E079", "E094", "E096", "E106", "E113"]
# TRAINING_SETUPS = list(itertools.product(individuals, tissues))

TRAINING_SETUPS = [('male_37', 'male_54', 'E065'),
 ('male_37', 'male_54', 'E079'),
 ('male_37', 'male_54', 'E094'),
 ('male_37', 'male_54', 'E096'),
 ('male_37', 'male_54', 'E106'),
 ('male_37', 'male_54', 'E113'),
 ('male_37', 'female_53', 'E065'),
 ('male_37', 'female_53', 'E079'),
 ('male_37', 'female_53', 'E094'),
 ('male_37', 'female_53', 'E096'),
 ('male_37', 'female_53', 'E106'),
 ('male_37', 'female_53', 'E113'),
 ('male_37', 'female_51', 'E065'),
 ('male_37', 'female_51', 'E079'),
 ('male_37', 'female_51', 'E094'),
 ('male_37', 'female_51', 'E096'),
 ('male_37', 'female_51', 'E106'),
 ('male_37', 'female_51', 'E113'),
 ('male_54', 'male_37', 'E065'),
 ('male_54', 'male_37', 'E079'),
 ('male_54', 'male_37', 'E094'),
 ('male_54', 'male_37', 'E096'),
 ('male_54', 'male_37', 'E106'),
 ('male_54', 'male_37', 'E113'),
 ('male_54', 'female_53', 'E065'),
 ('male_54', 'female_53', 'E079'),
 ('male_54', 'female_53', 'E094'),
 ('male_54', 'female_53', 'E096'),
 ('male_54', 'female_53', 'E106'),
 ('male_54', 'female_53', 'E113'),
 ('male_54', 'female_51', 'E065'),
 ('male_54', 'female_51', 'E079'),
 ('male_54', 'female_51', 'E094'),
 ('male_54', 'female_51', 'E096'),
 ('male_54', 'female_51', 'E106'),
 ('male_54', 'female_51', 'E113'),
 ('female_53', 'male_37', 'E065'),
 ('female_53', 'male_37', 'E079'),
 ('female_53', 'male_37', 'E094'),
 ('female_53', 'male_37', 'E096'),
 ('female_53', 'male_37', 'E106'),
 ('female_53', 'male_37', 'E113'),
 ('female_53', 'male_54', 'E065'),
 ('female_53', 'male_54', 'E079'),
 ('female_53', 'male_54', 'E094'),
 ('female_53', 'male_54', 'E096'),
 ('female_53', 'male_54', 'E106'),
 ('female_53', 'male_54', 'E113'),
 ('female_53', 'female_51', 'E065'),
 ('female_53', 'female_51', 'E079'),
 ('female_53', 'female_51', 'E094'),
 ('female_53', 'female_51', 'E096'),
 ('female_53', 'female_51', 'E106'),
 ('female_53', 'female_51', 'E113'),
 ('female_51', 'male_37', 'E065'),
 ('female_51', 'male_37', 'E079'),
 ('female_51', 'male_37', 'E094'),
 ('female_51', 'male_37', 'E096'),
 ('female_51', 'male_37', 'E106'),
 ('female_51', 'male_37', 'E113'),
 ('female_51', 'male_54', 'E065'),
 ('female_51', 'male_54', 'E079'),
 ('female_51', 'male_54', 'E094'),
 ('female_51', 'male_54', 'E096'),
 ('female_51', 'male_54', 'E106'),
 ('female_51', 'male_54', 'E113'),
 ('female_51', 'female_53', 'E065'),
 ('female_51', 'female_53', 'E079'),
 ('female_51', 'female_53', 'E094'),
 ('female_51', 'female_53', 'E096'),
 ('female_51', 'female_53', 'E106'),
 ('female_51', 'female_53', 'E113')]



def auprc(target, pred, **kwargs):
    precision, recall, thresholds = precision_recall_curve(target, pred)
    return auc(recall, precision)

def mse(y_true, y_pred):
    return np.square(y_true - y_pred).mean()


def gwcorr(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]



def main(args):
    # Load the dataset class for the Roadmap dataset to have access to precomputed gap masks
    # dataset = load_dataset("PredictdChr21")
    mask = np.load("data/ch21_mask.npy").astype(bool) #dataset._get_bin_mask(exclude_gaps=True)
    n_cells = 6
    n_assays = 5

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
    experiment_results_dir = "outputs/results_ENTEx_LOO_tissue_transfer/{}".format(args.experiment_name)
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
    with h5py.File(join(args.data_folder, "{}_r.hdf5".format(args.transfer_individual)), "r") as f:
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
    print(training_generator[0])
    print("Training generator available\n")
    model.fit(training_generator, callbacks=callbacks, epochs=args.epochs)#args.n_epochs)

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
    with h5py.File(join(args.data_folder, "{}_r.hdf5".format(args.target_individual)), "r") as f:
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
        # RunningMetricPrinter(log_freq=500),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='loss', mode='min'),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path, monitor='loss', mode='min', save_weights_only=True, save_best_only=True)
    ]
    model.fit(target_ft_generator, callbacks=ft_callbacks, epochs=args.ft_epochs)

    # Restoring best ft model
    model.load_weights(checkpoints_path)

    
    test_generator = ValInMemGenerator(
        transfer_data, transfer_cell_ids, transfer_assay_ids, target_data, test_cell_ids, test_assay_ids, transform='arcsinh')


    index = 0
    predicted_tracks = np.zeros_like(target_data)

    for i_batch in tqdm(range(len(test_generator))):
        batch, targets = test_generator[i_batch]
        batch_len = len(batch[0])
        # if i_batch==0:
        predictions = model(batch, training=False).numpy()
        # try:
        predicted_tracks[index:index+batch_len, :] = predictions
        # except:
        #     break
        index += batch_len
    if not os.path.exists(experiment_results_dir):
        os.makedirs(experiment_results_dir)
    np.save(join(experiment_results_dir, "trsf-{}_tgt-{}_tgttissue-{}_imputed.npy".format(
                                                    args.transfer_individual, args.target_individual, args.target_tissue)), predicted_tracks)


    print("\n\nEvaluating metrics..\n")

    # Removing BL regions
    predicted_tracks = predicted_tracks[blacklist_mask_no_gaps,:]
    target_data = np.arcsinh(target_data[blacklist_mask_no_gaps,:])

    genomewide_reconstruction = {"MSE Global": mse, "GW Corr": gwcorr}
    MACS_vs_bins_classif = {"AUPRC MACS": auprc}
    results_metrics = []
    for j, track_name in tqdm(enumerate(target_tracks)):
        for mname, mfun in genomewide_reconstruction.items():
            val = mfun(target_data[:, j], predicted_tracks[:, j])
            results_metrics.append(
                {"track": track_name, "transfer_individual": args.transfer_individual, 
                "target_individual": args.target_individual, 
                "category": "Genome-Wide Signal Reconstruction",
                    "predictor": "eDICE", "metric": mname, "value": val})

        obs_macs_mask = np.load("data/ENTEx/MACS_peaks/{}_{}_observed_raw.npy".format(
                args.target_individual, track_name))[blacklist_mask_no_gaps].astype(bool)
        for mname, mfun in MACS_vs_bins_classif.items():
            val = mfun(obs_macs_mask, predicted_tracks[:, j])
            results_metrics.append(
                {"track": track_name, "transfer_individual": args.transfer_individual, 
                "target_individual": args.target_individual, 
                "category": "Genome-Wide Signal Reconstruction",
                    "predictor": "eDICE", "metric": mname, "value": val})

    results_metrics = pd.DataFrame(results_metrics)
    results_metrics.to_csv(join(experiment_results_dir, "trsf-{}_tgt-{}_tgttissue-{}_metrics.csv".format(
                                                    args.transfer_individual, args.target_individual, args.target_tissue)), index=False)

    print("Analysis complete!")



MODEL_DEFAULTS = dict(layer_norm_type=None,
                        decoder_layers=2,
                        decoder_hidden=512,
                        decoder_dropout=0.3,
                        total_bins=None)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--training_setup", type=int, default=5)

    parser.add_argument("--data_folder", type=str,
                        default="data/ENTEx/processed_data")
    # parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--ft_learning_rate", type=float, default=0.00003)
    parser.add_argument("--experiment_name", default="test_debug")
    parser.add_argument("--experiment_group", default="LOO_tissue_transfer")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--ft_epochs", default=15, type=int)
    parser.add_argument("--blacklist_path", type=str, default="localdata/hg19-blacklist.v2.bed")


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
    
    # parser.add_argument('--n_targets', type=int, default=14)
    # parser.add_argument('--min_targets', type=int, default=None)
    # parser.add_argument('--max_targets', type=int, default=None)
    
    parser.add_argument('--checkpoint', action="store_true")
    # parser.add_argument('--resume', action="store_true",
    #                     help="resume training if an existing checkpoint is available")
    # parser.add_argument('--agg_metrics_only', action="store_true")
    parser.set_defaults(**MODEL_DEFAULTS)
    args = parser.parse_args()


    ts = TRAINING_SETUPS[args.training_setup]
    args.transfer_individual, args.target_individual, args.target_tissue = ts
    # args.transfer_individual = ind_pairings[args.target_individual]

    main(args)
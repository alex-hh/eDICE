import os
import tensorflow as tf
import argparse, os
import numpy as np

import time

from edice.data_loaders.custom_dataset import load_custom_dataset
from edice.models.model_utils import load_model
from edice.utils.train_utils import get_callbacks, find_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str)
    
    parser.add_argument('--dataset_filepath', default="roadmap/SAMPLE_chr21_roadmap_train.h5")
    parser.add_argument('--data_dir', default="sample_data")
    parser.add_argument('--idmap', default="sample_data/roadmap/idmap.json")
    parser.add_argument('--dataset_name', default="mySampleRoadmap")
    parser.add_argument('--split_file', type=str, default="sample_data/roadmap/predictd_splits.json")
    parser.add_argument('--gap_file', type=str, default="annotations/hg19gap.txt")
    parser.add_argument('--blacklist_file', type=str, default="annotations/hg19-blacklist.v2.bed")
    parser.add_argument('--track_resolution', type=int, default=25)                       
    
    parser.add_argument('--experiment_group', type=str, default=None)  
    parser.add_argument('--train_splits', type=str, default=["train"], nargs="+")

    
    parser.add_argument('--test_run', action="store_true")
    parser.add_argument('--seed', default=211, type=int)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--transformation', type=str, choices=["arcsinh"], default=None)
    
    parser.add_argument('--n_attn_heads', type=int, default=4)
    parser.add_argument('--n_attn_layers', type=int, default=1)
    
    
    parser.add_argument('--decoder_layers', type=int, default=2)
    parser.add_argument('--decoder_hidden', type=int, default=2048)
    parser.add_argument('--decoder_dropout', type=float, default=0.3)
    parser.add_argument('--total_bins', default=None)
    
    parser.add_argument('--layer_norm_type', default=None)

    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--intermediate_fc_dim', type=int, default=128)
    parser.add_argument('--transformer_dropout', type=float, default=0.1)
    parser.add_argument('--embedding_dropout', type=float, default=0.)
    parser.add_argument('--intermediate_fc_dropout', type=float, default=0.)
    
    parser.add_argument('--n_targets', type=int, default=120)
    parser.add_argument('--min_targets', type=int, default=None)
    parser.add_argument('--max_targets', type=int, default=None)
    
    parser.add_argument('--final_checkpoint', action="store_true")
    parser.add_argument('--resume', action="store_true",
                        help="resume training if an existing checkpoint is available")
    
    
    parser.add_argument('--agg_metrics_only', action="store_true")
    




    args = parser.parse_args()
    if args.min_targets is not None:
        assert args.min_targets >= 1 and args.max_targets
        args.n_targets = None
    else:
        assert args.n_targets, "must either specify n_targets or (min, max) targets"
    
    return args


def train_model(model, dataset, epochs, callbacks=None,
                checkpoint=None, start_epoch=0, n_targets=120,
                test_run=False, min_targets=None,
                per_track_metrics=False, max_targets=None,
                train_splits=["train"]):
    """
    Evaluate once per epoch, that's it
    """
    if checkpoint is not None:
        print("loading weights from checkpoint", checkpoint)
        model.load_weights(checkpoint)
    print(f"training for {epochs} epochs")
    # callbacks = callbacks or [RunningMetricPrinter(), EpochTimer()]
    # dataset_fit and dataset_evaluate handle setting up metrics
    model.dataset_fit(dataset, n_targets=n_targets,
                      min_targets=min_targets,
                      max_targets=max_targets,
                      epochs=epochs,
                      verbose=1 if test_run else 2,
                      callbacks=callbacks,
                      initial_epoch=start_epoch,
                      per_track_metrics=per_track_metrics,
                      train_splits=train_splits)


def main(args):
    print(args)
    print('SETTING SEED', args.seed)
    np.random.seed(args.seed)
    
    dataset = load_custom_dataset(total_bins=1000 if args.test_run else None,
                           splits=args.split_file,
                           filepath=args.dataset_filepath,
                           idmap=args.idmap,
                           name=args.dataset_name,
                           data_dir=args.data_dir
                           )
    
    n_cells, n_assays = len(dataset.cells), len(dataset.assays)
    print("Loading model", flush=True)
    model = load_model(n_cells, n_assays, args, compile_model=True)
    # could add to on_train_begin via callback but then can't control start epoch...
    if args.resume:
        checkpoint, start_epoch = find_checkpoint(args.experiment_name,
                                                  args.experiment_group)
        # TODO - check all arguments match
    else:
        print("Training from scratch (checkpoint None)")
        checkpoint = None
        start_epoch = 0

    print("Getting callbacks", flush=True)
    callbacks, output_dir = get_callbacks(args, dataset, checkpoint=checkpoint)
    train_model(model, dataset, epochs=args.epochs, callbacks=callbacks, 
                test_run=args.test_run, checkpoint=checkpoint,
                start_epoch=start_epoch, n_targets=args.n_targets,
                min_targets=args.min_targets, max_targets=args.max_targets,
                per_track_metrics=not args.agg_metrics_only,
                train_splits=args.train_splits)

    model.dataset_predict(
        dataset,
        support_splits=args.train_splits,
        val_split="test",
        exclude_gaps=False,
        exclude_blacklist=False,
        save_preds=True,
        outfile=os.path.join(output_dir, f"{args.dataset_name}_test_preds.npz"),
    )

    if args.final_checkpoint:
        print("saving final weights", flush=True)
        model.save_weights(os.path.join(output_dir, "/checkpoints/train_end"))


if __name__ == '__main__':
    start_time = time.time()

    args = parse_args()
    main(args)

    print("Script execution time: {} minutes".format((time.time()-start_time)/60))
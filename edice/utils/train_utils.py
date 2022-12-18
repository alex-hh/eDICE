import argparse
import os
import json
import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from edice.data_loaders.metadata import read_splits_json
from edice.utils.callbacks import RunningMetricPrinter, EpochTimer, AssayAverager
from edice.utils.CONSTANTS import OUTPUT_DIR


class ConfigSaver(tf.keras.callbacks.Callback):

    """
    Use a callback so that config only gets saved if 
    we get to on train begin
    """

    def __init__(self, config, filepath, splits=None, format="yaml"):
        self.config = config
        self.splits = splits
        self.filepath = filepath
        self.splits_path = os.path.join(os.path.dirname(self.filepath), "splits.json")
        self.format = format

    def on_train_begin(self, logs):
        save_config(self.config, self.filepath, format=self.format)
        if self.splits is not None:
            save_config(self.splits, self.splits_path, format="json")


def get_callbacks(args, dataset, checkpoint=None):
    output_dir = get_output_dir(args.experiment_name, args.experiment_group,
                                resume=args.resume)
    splits = read_splits_json(args.split_file)

    if args.resume:
        checkpoint_code = os.path.basename(checkpoint)  # epoch-loss
        final_metrics_file = f"final_metrics_resume_{checkpoint_code}.csv"
    else:
        final_metrics_file = "final_metrics.csv"
    callbacks = [   
                    # log_freq units are currently batch size: 390 matches my old log_freq
                    RunningMetricPrinter(log_freq=3 if args.test_run else 390),
                    EpochTimer(),
                    AssayAverager(dataset.assays),
    ]
    if not args.test_run:
        callbacks += [
            CSVLogger(output_dir / "train_log.csv", append=args.resume),
            # https://www.tensorflow.org/guide/keras/save_and_serialize
            ModelCheckpoint(
                output_dir / "checkpoints/edice",
                save_best_only=True,
                monitor="val_loss",
                # defaults true anyway for subclassed models
                # https://github.com/tensorflow/tensorflow/blob/b36436b087bd8e8701ef51718179037cccdfc26e/tensorflow/python/keras/callbacks.py#L1219
                # model._is_graph_network tests whether model is functional
                save_weights_only=True),
            ]
        if not args.resume:
            callbacks.append(ConfigSaver(
                args, output_dir / "config.yaml",
                splits=splits, format="yaml"))

    return callbacks, output_dir


def save_config(config, filepath, format="yaml"):
    if isinstance(config, argparse.Namespace):
        config = vars(config)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as outfile:
        if format == "yaml":
            yaml.dump(config, outfile)
        elif format == "json":
            json.dump(config, outfile, indent=4)


def load_saved_config(experiment_name, experiment_group=None, format="yaml"):
    output_dir = get_output_dir(experiment_name, experiment_group=experiment_group,
                                resume=True)

    print(f"Loading saved config from {str(output_dir)}")
    
    if format == "yaml":
        with open(output_dir / "config.yaml", "r") as yf:
            return yaml.load(yf, Loader=yaml.FullLoader)


def find_last_version(output_dir):
    existing_versions = []
    if os.path.isdir(output_dir):
        for d in os.listdir(output_dir):
            # first branch of this is we're not interested in files
            subdir = os.path.join(output_dir, d)
            if os.path.isdir(subdir) and d.startswith("version_"):
                existing_versions.append(int(d.split("_")[1]))
    if existing_versions:
        return max(existing_versions)
    else:
        return 0


def get_output_dir(experiment_name, experiment_group=None,
                   resume=False, add_version=True, version=None,
                   output_dir=None):
    """
    Versioning is baked in by default as in pytorch lightning (based on pytorch lightning code)
    """
    output_dir = (output_dir or OUTPUT_DIR) / (experiment_group or "") / experiment_name
    if add_version:
        last_version = find_last_version(output_dir)
        version = version or (last_version if resume else last_version + 1)
        output_dir /= f"version_{version}"
    if resume:
        assert os.path.isdir(output_dir),\
        f"Cannot resume because output directory {output_dir} does not exist"
    else:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def find_latest_checkpoint_in_dir(checkpoints_dir):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoints_dir)
    if latest_checkpoint is None:
        print(f"No checkpoints found in dir {checkpoints_dir}, starting from scratch")
        start_epoch = 0
    else:
        print("Resuming from checkpoint", latest_checkpoint)
        start_epoch = int(os.path.basename(latest_checkpoint).split("-")[0])

    return latest_checkpoint, start_epoch


def find_checkpoint(experiment_name, experiment_group=None, version=None):
    """
    Check for a checkpoint if available
    use checkpoint manager https://www.tensorflow.org/api_docs/python/tf/train/CheckpointManager
    """
    experiment_dir = get_output_dir(experiment_name,
                                    experiment_group=experiment_group,
                                    resume=True,
                                    version=version)
    checkpoints_dir = experiment_dir / "checkpoints"
    return find_latest_checkpoint_in_dir(checkpoints_dir)

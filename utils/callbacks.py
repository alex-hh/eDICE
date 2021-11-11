import csv
import json
import os
import re
import yaml
from collections import defaultdict
from time import time
import tensorflow as tf


class Checkpoint(tf.keras.callbacks.ModelCheckpoint):

    def __init__(self, filepath, n_checkpoints_max=5, **kwargs):
        self.n_checkpoints_max = n_checkpoints_max
        super().__init__(filepath, **kwargs)

    def _save_model(self, epoch, logs):
        self.cleanup()
        super()._save_model(epoch, logs)

    def cleanup(self):
        """
        based on ModelCheckpoint._get_most_recently_modified_file_matching_pattern
        """
        print("filepath", self.filepath)
        dir_name = os.path.dirname(self.filepath)
        base_name = os.path.basename(self.filepath)
        base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'
        chckpoints = []
        filepaths = defaultdict(list)
        if os.path.isdir(dir_name):
            for file_name in os.listdir(dir_name):
                # Only consider if `file_name` matches the pattern.
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    chckpoints.append(os.path.basename(file_path))
                    filepaths[os.path.basename(file_path)].append(file_path)
        
        chckpoints = list(set(chckpoints))
        print("Matched chckpoints", chckpoints)
        chckpoints = sorted(chckpoints, key=lambda f: os.path.getmtime(filepaths[f][-1]),
                            reverse=True)
        chckpoints_to_remove = chckpoints[self.n_checkpoints_max-1:]  # a checkpoint is about to be added
        for chckpoint in chckpoints_to_remove:
            for f in filepaths[chckpoint]:
                os.remove(f)


class EpochTimer(tf.keras.callbacks.Callback):
  # just make sure to include before csvlogger

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time() - self.epoch_time_start
        logs['time'] = epoch_time


class AssayAverager(tf.keras.callbacks.Callback):

    """
    Compute averages of metrics averaged over assays
        i.e. if assay level metrics are tracking
        average mse / correlation for all tracks from a 
        given assay,
        this callback will additionally track the average
        of those averages
    """

    def __init__(self, assays, metric_prefix="assayAvg"):
        self.metric_prefix = metric_prefix
        self.assays = assays
        super().__init__()

    def is_assay_avg_metric(self, metric):
        name_index = 0
        if metric.startswith("val_"):
            name_index = 1
        return metric.split("_")[name_index] in self.assays

    def compute_assay_averages(self, logs=None):
        if logs is None:
            logs = {}
        metric_totals = defaultdict(int)
        metric_counts = defaultdict(int)
        val = False
        for k, v in logs.items():
            # c.f. tracksubsetmetricmixin which constructs name as
            # name='_'.join([subset_name or "", metric_name])
            name_parts = k.split("_")
            if self.is_assay_avg_metric(k):
                if k.startswith("val_"):
                    val = True
                    name_parts = name_parts[1:]
                assay, metric = name_parts[0], name_parts[1]
                metric_totals[metric] += v
                metric_counts[metric] += 1
        assayavg_base = ("val_" if val else "") + self.metric_prefix + "_"
        assay_averages = {assayavg_base + m: tot / metric_counts[m]
                          for m, tot in metric_totals.items()}
        # TODO: figure out why these aren't being printed in the progbar?
        print("Metrics averaged across assays:\n")
        print("\t" + " - ".join([f"{k}: {v:.4f}" for k, v in assay_averages.items()]))
        logs.update(assay_averages)
        return logs

    def on_epoch_end(self, epoch, logs):
        logs = self.compute_assay_averages(logs)


class RunningMetricPrinter(tf.keras.callbacks.Callback):
  
    """
    Extends base logger to allow printing the loss multiple times per epoch 
    (every log_freq batches) instead of once per epoch (what BaseLogger does).

    Note that in tf2 the behaviour is modified to accommodate stateful metrics
    These are metrics which automatically compute running totals.

    c.f. my callback for neural processes
    """

    def __init__(self, log_freq, **kwargs):
        self.log_freq = log_freq
        super().__init__(**kwargs)

    def on_train_batch_end(self, batch, logs):
        # what tensorboard uses c.f. make_train_function
        counter = self.model._train_counter.numpy()
        should_log = self.model._train_counter % self.log_freq == 0

        if should_log:
            msg = f"Running averages ({counter} steps):  "
            msg += "  -  ".join(
                [f"{k}: {v:.4f}" for k, v in logs.items()]
            )
            print(msg, flush=True)

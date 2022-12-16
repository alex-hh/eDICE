import gc
import math
import os
import tensorflow as tf
import numpy as np

from utils import transforms, val_utils
from data_loaders.data_generators import format_inputs
from data_loaders import hdf5_utils


class BaseModel(tf.keras.Model):

    """
    Builtin keras model has default handling of metrics, loss, optimizer
        which are all defined in compile and applied in train/test/predict steps

    In order to customise this behaviour we need to subclass.
    The purpose of this BaseModel class is to handle metrics differently.

    So when calling model.compile for anything inheriting from this BaseModel class, 
    an optimizer and a builtin loss should be compiled, but (probably) no metrics.
    
    During training the dimensions of the model's outputs do
    not have fixed meaning (and in principle the number of dimensions 
        can vary)
    whereas, during validation, each dimension at each data point has 
    the same meaning corresponding to a single track.

    We therefore require the ability to add metrics to the model which
    only have meaning at validation time.

    Furthermore, we want to be able to compute such metrics
    on raw rather than transformed data.

    To do this this model has a setup_metrics method, separate to compile
    - which is where metrics to be handled in standard way are defined -
    where the metrics to be used are hardcoded for now.

    Importantly, the setup_metrics method takes a dataset as
    an argument, allowing metrics to be constructed which
    depend on the validation dataset: in particular the number
    of tracks (i.e. number of output dimensions of the model),
    and the identities of those tracks (so that we can compute
        averages over individual assay types)

    The setup_metrics method should be called before training
    or evaluating on a new dataset. To handle this automatically,
    this class provides dataset_fit and dataset_evaluate methods
    which wrap the metric setup phase together with the call to fit.

    Once metrics have been setup, the train/test/predict step methods
    update them as well as any metrics defined in the standard way in compile,
    returning stateful metric results in output dict from train_step 
    and test_step for compatibility with callbacks (access via logs).

    In this way both metrics defined in the standard way via compile,
    and in the dataset dependent way via setup_metrics will be accessible
    (as running averages) in the callback logs. The only difference
    in the current treatment is that y_true and y_pred are transformed
    back to raw values before metrics defined via setup metrics are
    computed, whereas compiled metrics are computed on the transformed
    values.
    """

    def __init__(self, transformation=None, dataset=None,
                 fixed_inputs=False, **kwargs):
        super().__init__(**kwargs)
        # reinstate class attribute (wiped off by framework?)
        self.input_names = getattr(self.__class__, "input_names", None)
        self.transformation = transformation
        self.fixed_inputs = fixed_inputs

    def reset_metrics(self):
        super().reset_metrics()  # I think any metric set as an attribute is automatically added to self.metrics? see _setup_metrics docstring

    # THESE METHODS HANDLE METRIC SETUP GIVEN A DATASET OBJECT
    # WHICH HAS A SPLITS ATTRIBUTE WHICH DETERMINES THE TRAIN/VAL SPLIT
    def dataset_fit(self,
                    dataset,
                    n_targets,
                    min_targets=None,
                    grouped_assay_metrics=True,
                    per_track_metrics=False,
                    max_targets=None,
                    val_split="val",
                    train_splits=["train"],
                    shuffle=True,
                    **fit_kwargs):
        """
        fit on the train split of the passed Dataset and evaluate on the val split
            use the Dataset's track mapping to setup val metrics
        """
        self.setup_metrics(dataset=dataset,
                           val_split=val_split,
                           grouped_assay_metrics=grouped_assay_metrics,
                           per_track_metrics=per_track_metrics)
        train_tracks = [t for split in train_splits for t in dataset.splits[split]]
        train_generator = dataset.get_train_generator(transform=self.transformation,
                                                      train_tracks=train_tracks,
                                                      n_targets=n_targets,
                                                      min_targets=min_targets,
                                                      max_targets=max_targets,
                                                      shuffle=shuffle,
                                                      fixed_inputs=self.fixed_inputs)
        val_generator = dataset.get_val_generator(transform=self.transformation,
                                                  fixed_inputs=self.fixed_inputs)
        self.fit(train_generator, validation_data=val_generator, **fit_kwargs)

    def dataset_evaluate(self,
                         dataset,
                         support_splits=None,
                         val_split="val",
                         grouped_assay_metrics=True,
                         per_track_metrics=False,
                         callbacks=None,
                         exclude_gaps=False,
                         exclude_blacklist=False,
                         **eval_kwargs):
        """
        evaluate on the val split of the passed Dataset
            use the Dataset's track mapping to setup val metrics
        """
        # n.b. metrics depend on targets only not supports
        self.setup_metrics(dataset,
                           val_split=val_split,
                           grouped_assay_metrics=grouped_assay_metrics,
                           per_track_metrics=per_track_metrics)
        val_generator = dataset.get_val_generator(
            support_splits=support_splits, target_split=val_split,
            transform=self.transformation, exclude_gaps=exclude_gaps,
            exclude_blacklist=exclude_blacklist, fixed_inputs=self.fixed_inputs,
        )
        return self.evaluate(val_generator, callbacks=callbacks, **eval_kwargs)

    def dataset_predict(self,
                        dataset,
                        support_splits=None,
                        val_split="val",
                        callbacks=None,
                        exclude_gaps=False,
                        exclude_blacklist=False,
                        save_preds=False,
                        outfile=None,
                        **eval_kwargs):
        val_generator = dataset.get_val_generator(
            support_splits=support_splits, target_split=val_split,
            transform=self.transformation, exclude_gaps=exclude_gaps,
            exclude_blacklist=exclude_blacklist, fixed_inputs=self.fixed_inputs,
        )
        target_tracks = dataset.splits[val_split]
        preds = self.predict(val_generator, callbacks=callbacks, **eval_kwargs)
        if save_preds:
            np.savez(outfile, preds)
        
        return preds

    def set_test_type(self, test_time=False):
        if test_time:
            self.test_type = "test"
        else:
            self.test_type = "val"

    # TODO: find a way to allow passing of support tracks and val tracks
    def setup_metrics(self, dataset, val_split="val", 
                      val_targets=None, grouped_assay_metrics=True, 
                      per_track_metrics=False, test_time=False):
        """
        val_metrics gives option to manually construct val_metrics 
        otherwise they will be inferred from dataset
        former is used in run_evaluation currently (but dataset_evaluate
            could probably be used instead)

        train metrics are pretty redundant tbh:
            the individual dimensions in our model do not have stable meanings
            so the per-dim metrics simply don't mean anything

        N.B. metrics added as attributes to the model or to containers
            which are attributes of the model are automatically tracked in
            self.metrics (via the l._metrics attribute which is accessed
                for the Model in self.metrics for l in self._flatten_layers)
            Q. does it matter whether this happens before or after compile?
            Q. where does this happen and is there any way of controlling it - 
                I think it impacts the saving and reloading of models (see below);
                we seem to arrive at errors if we add metrics with different shapes.

        I'm worried that metrics are trackables and get loaded with checkpoint
          there is no reason to save metrics (can I make them not trackable?)
        def _gather_saveables_for_checkpoint
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#save_weights
        When saving in TensorFlow format, all objects referenced by the network are saved in the 
        same format as tf.train.Checkpoint, including any Layer instances or Optimizer 
        instances assigned to object attributes.
        If by_name is True, weights are loaded into layers only if they share
        the same name. This is useful for fine-tuning or transfer-learning 
        models where some of the layers have changed.
        Only topological loading (by_name=False) is supported when 
        loading weights from the TensorFlow format. 
        """
        self.set_test_type(test_time)
        self.train_metrics, self.val_metrics, self.test_metrics = [], [], []

        if val_targets is None:
            val_metrics = dataset.get_val_metrics(
                val_split=val_split,
                grouped_assay_metrics=grouped_assay_metrics,
                per_track_metrics=per_track_metrics)
        else:
            val_metrics = dataset.get_targets_metrics(
                val_targets,
                grouped_assay_metrics=grouped_assay_metrics,
                per_track_metrics=per_track_metrics)

        if self.test_type == "val":
            self.val_metrics = val_metrics
        elif self.test_type == "test":
            self.test_metrics = val_metrics
        else:
            raise ValueError("test type must be either val or test")

    def transform_inputs(self, y):
        return transforms.get_tf_transformation(self.transformation)(y)

    def inv_transform_preds(self, y_pred):
        return transforms.get_inverse_tf_transformation(self.transformation)(y_pred)

    # https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
    def train_step(self, data):
        """
        Assumes that a standard builtin loss has been passed to compile

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        # ensures data is a tuple (x, y, sample_weight)
        # if the generator or data fed to fit does not include y/sample_weight
        # then the corresponding elements in the tuple will just be None
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        
        if isinstance(x, dict):
            # actually doesnt work with fit (why?)
            inputs = format_inputs(x)
        else:
            inputs = x

        # compute loss and gradients on transformed vals
        with tf.GradientTape() as tape:
            y_pred = self(inputs, training=True)
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss as setup in compile)
        self.compiled_metrics.update_state(y, y_pred)
        
        # inv transform inputs and preds to compute metrics on raw vals
        metrics = self.get_raw_metrics(y, y_pred, "train")
        # include built in metrics computed on transformed vals (i.e. loss)
        metrics.update({m.name: m.result() for m in self.compiled_loss.metrics})
        metrics.update({m.name: m.result() for m in self.compiled_metrics.metrics})

        return metrics

    def predict_step(self, data):
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        if isinstance(x, dict):
            inputs = [x['supports'], x['support_cell_ids'], x['support_assay_ids'],
                      x['target_cell_ids'], x['target_assay_ids']]
        else:
            inputs = x
        return self(inputs, training=False)

    def test_step(self, data):
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        
        if isinstance(x, dict):
            inputs = [x['supports'], x['support_cell_ids'], x['support_assay_ids'],
                      x['target_cell_ids'], x['target_assay_ids']]
        else:
            inputs = x
        
        # compute loss on transformed vals
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics. I think this might work despite the difference in size
        # between train and val n outputs - if it's just an average...
        self.compiled_metrics.update_state(y, y_pred)

        # compute metrics on raw vals
        metrics = self.get_raw_metrics(y, y_pred, self.test_type)
        # print("raw metrics", metrics)
        metrics.update({m.name: m.result() for m in self.metrics})
        return metrics

    def get_specified_metrics(self, y, y_pred, metrics, inv_transform=False):
        if inv_transform:
            y_pred = self.inv_transform_preds(y_pred)
            y = self.inv_transform_preds(y)

        for m in metrics:
            m.update_state(y, y_pred)

        metrics = {m.name: m.result() for m in metrics}
        return metrics

    # TODO add a get_transformed_metrics method (and add in setup_metrics)
    def get_raw_metrics(self, y_transformed, y_transformed_pred, mode):
        """
        Given y and y_pred (possibly on some transformed scale)
        Inverse transforms onto the log10 p value ('raw') scale
        And computes metrics on this scale

        Note that the metric classes for train and val are different
        because the outputs have different shapes during train and val
        (different numbers of dimensions)
        """

        if mode == 'train':
            metrics = getattr(self, "train_metrics", [])
        elif mode == 'val':
            metrics = getattr(self, "val_metrics", [])
        elif mode == 'test':
            metrics = getattr(self, "test_metrics", [])
        else:
            raise ValueError("Mode must be either train, val or test")
        
        return self.get_specified_metrics(
            y_transformed, y_transformed_pred, metrics, inv_transform=True)

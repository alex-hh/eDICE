"""
A notable feature of these metrics is that their state is multi-dimensional
With the size of the state variable depending on the shape of inputs

To handle this we follow what is done in the AUC metric and defer
the creation of the weights storing any running totals until the first time
the metric is called
"""
import collections
import re
import tensorflow as tf
from tensorflow.keras import backend as K
from edice.utils import transforms


METRIC_NAMES = ["mse", "pearsonr", "auc", "auprc", "precision", "recall", "spearmanr", "loss"]
MetricModifiers = collections.namedtuple(
    "MetricModifiers", ["transformation", "track_name", "assay_name",
                        "peak_thresholds_name", "metric_name"]
    )


def is_peak_thresholds_name(name_part):
    return bool(re.match("gt\d+Preds", name_part))


def metric_modifiers_from_name(metric_name, dataset, **modifiers):
    """
    Convention is that name parts (modifiers and name of base metric)
    are separated by an underscore.

    We have to use this recursive method to parse because if any name
    part is None it will not appear in the name, so name parts do not
    have fixed semantics (the more modifiers there are, the more
        name parts there are.)

    An alternative to parsing would be just requiring metrics
    with the same meaning to have exactly matching names

    The motivation for parsing is that it makes it more straightforward
    to do grouping by specific parts of the metric (e.g. in order
    to calculate per track or per assay metric averages),
    which will be necessary for analysis and useful for printing
    verbose summaries from evaluation scripts
    """
    name_parts = metric_name.split("_")
    if len(name_parts) == 1:
        metric_name = name_parts[-1]
        assert metric_name in METRIC_NAMES,\
            (f"last part of metric name should be one of {METRIC_NAMES}"
             f" but is {metric_name} ({name_parts})")
        all_modifiers = {m: None for m in ["transformation", "track_name",
                                           "assay_name", "peak_thresholds_name"]}
        all_modifiers.update(modifiers)
        return MetricModifiers(metric_name=metric_name,
                               **all_modifiers)
    elif name_parts[0] in transforms.TRANSFORMATIONS:
        modifiers["transformation"] = name_parts[0]
    elif name_parts[0] in dataset.tracks:
        modifiers["track_name"] = name_parts[0]
    elif name_parts[0] in dataset.assays:
        modifiers["assay_name"] = name_parts[0]
    elif is_peak_thresholds_name(name_parts[0]):
        modifiers["peak_thresholds_name"] = name_parts[0]
    else:
        raise ValueError(f"Unrecognized name part: {name_parts[0]} ({metric_name})")
    metric_name = "_".join(name_parts[1:])

    return metric_modifiers_from_name(metric_name, dataset, **modifiers)


class MetricResult:

    def __init__(self, metric_name, value,
                 transformation, track_name,
                 assay_name, peak_thresholds_name):
        self.metric_name = metric_name
        self.value = float(value)
        self.transformation = transformation
        self.track_name = track_name
        self.assay_name = assay_name
        self.peak_thresholds_name = peak_thresholds_name

    @classmethod
    def from_key_val(cls, key, val, dataset):
        mods = metric_modifiers_from_name(key, dataset)
        return cls(**mods._asdict(), value=val)

    @property
    def modified_metric_name(self):
        parts = [self.transformation, self.peak_thresholds_name,
                 self.metric_name]
        return "_".join([part for part in parts if part is not None])

    @property
    def aggregation_level(self):
        if self.track_name is not None:
            assert self.assay_name is None
            return "track"
        elif self.assay_name is not None:
            return "assay"
        else:
            return None

    @property
    def group_id(self):
        if self.track_name is not None:
            return self.track_name
        elif self.assay_name is not None:
            return self.assay_name
        else:
            return "global"

    def __repr__(self):
        s = f"{self.group_id} {self.modified_metric_name}: {self.value:.2f}"
        return s


# TODO - incorporate into metricresult, not sure its used atm?
def build_metric_name(metric_name, track_name=None,
                      assay_name=None, peak_thresholds_name=None,
                      transformation=None):
    if track_name is not None:
        assert assay_name is None
    parts = [transformation, track_name, assay_name, metric_name]
    metric_name = "_".join([part for part in parts if part is not None])
    return metric_name


class PerDimMetric(tf.keras.metrics.Metric):
    """
    Unlike the base reduction metric, allow for tracking variables and 
    calculating metrics on each dimension individually - 
    only averaging over the dimensions in self.result, (whereas 
        tf.keras.metrics.Reduction averages over dimensions earlier,
        storing scalar total and count state variables rather than
        per-dim state variables)
    to provide an average of per-dim metrics
    """

    def reset_state(self):
        """Resets all of the metric state variables.
        This function is called between epochs/steps,
        when a metric is evaluated during training.

        Batch set value just sets the values of many
        tensors at once
        """
        zeros = K.batch_get_value([K.zeros_like(v) for v in self.variables])
        K.batch_set_value([(v, zeros) for v, zeros in zip(self.variables, zeros)])

    def result(self):
        return tf.reduce_mean(self.per_dim_result())


class TrackSubsetMetricMixin(tf.keras.metrics.Metric):

    """
    Return average of the metric computed over a subset of the tracks
        relevant subsets are e.g. all belonging to an individual assay
    """

    def __init__(self, track_ids, metric_name, subset_name=None, **kwargs):
        self.track_ids = track_ids
        metric_name = '_'.join([subset_name or "", metric_name])
        super().__init__(len(track_ids), name=metric_name,
                         **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.gather(y_true, self.track_ids, axis=-1),
                                    tf.gather(y_pred, self.track_ids, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({'track_ids': self.track_ids})
        return config


class MeanSquaredError(PerDimMetric):

    """
    Mean squared error, averaged across tracks

    Refs:
     AUC metric builtin

     N.B. In v2, AUC should be initialized outside of any tf.functions, and therefore in
            eager mode.
    """

    def __init__(self, output_dim, name='mse', **kwargs):
        super().__init__(name=name, **kwargs)

        # sum of squared errors, summing each dimension separately
        self.sse = self.add_weight('sse', shape=(output_dim,),
                                   initializer='zeros')
        self.count = self.add_weight('count', shape=(),
                                     initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if tf.rank(y_true) == 1:
            y_true = tf.expand_dims(y_true, -1)
        if tf.rank(y_pred) == 1:
            y_pred = tf.expand_dims(y_pred, -1)

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # print(K.eval(K.shape(y_true)))
        sse = tf.reduce_sum(tf.math.square(y_pred - y_true), axis=0)

        # tf.shape is a tensor, tensor.shape is a tensorshape. the former is to be preferred
        # for dynamic computations (c.f. tf.tensor docs); tensor.shape is defined at construction
        # time and may be unknown; tf.shape is always known - for batch size using tf.shape is crucial
        # must cast to float because we will divide a float by this count later
        # and tf does not do automatic type casting
        num_values = tf.cast(tf.shape(y_true)[0], tf.float32)
        # print("Adding ", num_values, " to current count", self.count)
        self.count.assign_add(num_values)
        self.sse.assign_add(sse)

    def per_dim_result(self):
        # print("Overall count", self.count)
        # print("SSEs", self.sse)
        return self.sse / self.count


class AssayLevelMSE(TrackSubsetMetricMixin, MeanSquaredError):

    def __init__(self, track_ids, metric_name="mse",
                 subset_name=None, **kwargs):
        super().__init__(track_ids, metric_name=metric_name,
                         subset_name=subset_name, **kwargs)


class TrackLevelMSE(TrackSubsetMetricMixin, MeanSquaredError):

    def __init__(self, track_id, metric_name="mse",
                 track_name=None, **kwargs):
        super().__init__([track_id], metric_name=metric_name,
                         subset_name=track_name, **kwargs)


class PearsonCorrelation(PerDimMetric):

    """
    Compute the pearson correlation between y_true and y_pred
    For y_true and y_pred with shapes (batch_size, output_dim)
    Returns the mean of the correlations for each dimension

    Refs:

        https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras/46620771
        https://github.com/tensorflow/tensorflow/blob/23c218785eac5bfe737eec4f8081fd0ef8e0684d/tensorflow/contrib/metrics/python/ops/metric_ops.py#L3090
    """

    def __init__(self, output_dim, name='pearsonr', **kwargs):
        super().__init__(name=name, **kwargs)

        # product of ytrue and ypred
        self.yy_totals = self.add_weight('yy_total', shape=(output_dim,),
                                         initializer='zeros')
        self.ypred_totals = self.add_weight('ypred_total', shape=(output_dim,),
                                            initializer='zeros')
        self.ytrue_totals = self.add_weight('ytrue_total', shape=(output_dim,),
                                            initializer='zeros')
        self.ypred_sq_totals = self.add_weight('ypred_sq_total', shape=(output_dim,),
                                               initializer='zeros')
        self.ytrue_sq_totals = self.add_weight('ytrue_sq_total', shape=(output_dim,),
                                               initializer='zeros')
        self.count = self.add_weight('count', shape=(), initializer='zeros')
        # self._built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        if tf.rank(y_true) == 1:
            y_true = tf.expand_dims(y_true, -1)
        if tf.rank(y_pred) == 1:
            y_pred = tf.expand_dims(y_pred, -1)

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # if not self._built:
            # self._build(y_true.shape)

        # tf multiply gives elementwise product. this is simpler than worrying about shapes for matmul/dot
        sum_yy = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=0)
        sum_ypred_sq = tf.reduce_sum(tf.math.square(y_pred), axis=0)
        sum_ytrue_sq = tf.reduce_sum(tf.math.square(y_true), axis=0)
        sum_ypred = tf.reduce_sum(y_pred, axis=0)
        sum_ytrue = tf.reduce_sum(y_true, axis=0)
        # tf.shape is a tensor, tensor.shape is a tensorshape. the former is to be preferred
        # for dynamic computations (c.f. tf.tensor docs); tensor.shape is defined at construction
        # time and may be unknown; tf.shape is always evaluated at execution time - for batch size using tf.shape is crucial
        # must cast to float because we will divide a float by this count later
        # and tf does not do automatic type casting
        num_values = tf.cast(tf.shape(y_true)[0], dtype=tf.float32)

        self.yy_totals.assign_add(sum_yy)
        self.ypred_totals.assign_add(sum_ypred)
        self.ytrue_totals.assign_add(sum_ytrue)
        self.ypred_sq_totals.assign_add(sum_ypred_sq)
        self.ytrue_sq_totals.assign_add(sum_ytrue_sq)
        self.count.assign_add(num_values)

    def per_dim_result(self):
        mus_ypred = self.ypred_totals / self.count
        mus_ytrue = self.ytrue_totals / self.count
        covs = (self.yy_totals / self.count) - tf.multiply(mus_ytrue, mus_ypred)
        sds_ypred = tf.math.sqrt((self.ypred_sq_totals/self.count) \
                                 - tf.math.square(mus_ypred))
        sds_ytrue = tf.math.sqrt((self.ytrue_sq_totals/self.count) \
                                 - tf.math.square(mus_ytrue))
        return covs / tf.multiply(sds_ypred, sds_ytrue)


class AssayLevelCorr(TrackSubsetMetricMixin, PearsonCorrelation):
    
    def __init__(self, track_ids, metric_name="pearsonr",
                 subset_name=None, **kwargs):
        super().__init__(track_ids, metric_name=metric_name,
                         subset_name=subset_name, **kwargs)


class TrackLevelCorr(TrackSubsetMetricMixin, PearsonCorrelation):

    def __init__(self, track_id, metric_name="pearsonr",
                 track_name=None, **kwargs):
        super().__init__([track_id], metric_name=metric_name,
                         subset_name=track_name, **kwargs)

import numpy as np
import tensorflow as tf

try:
    from sklearn import metrics as sklearn_metrics
    SKLEARN_IS_AVAILABLE = True
except ImportError:
    SKLEARN_IS_AVAILABLE = False


def _sklearn_auc(targets, preds, track_names, peak_threshold=0.01):
    results = {}
    pct_threshold = int(100*peak_threshold)
    peak_threshold = np.percentile(targets, 100 - pct_threshold,
                                   axis=0)
    peak_targets = (targets > peak_threshold).astype(int)
    # print(f"Shapes: peak targets {peak_targets.shape} preds {preds.shape}")
    # print(f"nans: {np.sum(np.argwhere(~np.isfinite(peak_targets)))}"
    #       f" {np.sum(np.argwhere(~np.isfinite(preds)))}")

    for (track_i, track_name) in enumerate(track_names):
        target_track_peaks = peak_targets[:, track_i]
        pred_track_full = preds[:, track_i]
        auc_name = f"{track_name}_gt{pct_threshold}PredsFull_auc"
        auprc_name = f"{track_name}_gt{pct_threshold}PredsFull_auprc"
        results[auc_name] = sklearn_metrics.roc_auc_score(
            target_track_peaks, pred_track_full)
        results[auprc_name] = sklearn_metrics.average_precision_score(
            target_track_peaks, pred_track_full)
    return results


def _keras_auc(targets, preds, track_names, peak_threshold=0.01):
    """
    N.B. this is very slow with default num_thresholds.
    Whereas sklearn is fast and seems to use many more thresholds
    """
    results = {}
    # AUC: "For best results, predictions should be distributed approximately"
    # uniformly in the range [0, 1] and not peaked around 0 or 1.
    # TODO: compare to sklearn roc auc score?
    pct_threshold = int(100*peak_threshold)
    peak_threshold = np.percentile(targets, 100 - pct_threshold,
                                   axis=0)
    peak_targets = (targets > peak_threshold).astype(int)
    # rescale preds to [0,1]
    preds = (preds - preds.min(axis=0)) / (preds.max(axis=0) - preds.min(axis=0))
    auc = tf.keras.metrics.AUC()
    auprc = tf.keras.metrics.AUC(curve="PR", name="auprc")
    # https://davemcg.github.io/post/are-you-in-genomics-stop-using-roc-use-pr/

    for (track_i, track_name) in enumerate(track_names):
        # TODO: test whether this can depend on transformation?
        # and compare to sklearn (which uses trapezoidal rather than integral sums):
        # https://www.khanacademy.org/math/ap-calculus-ab/ab-integration-new/ab-6-2/v/trapezoidal-approximation-of-area-under-curve
        # (it is an approximation of the integral via a Riemann sum.)
        # sklearn seems to be much faster...
        target_track_peaks = peak_targets[:, track_i]
        pred_track_full = preds[:, track_i]
        auc.reset_states()
        auprc.reset_states()
        auc.update_state(target_track_peaks, pred_track_full)
        auprc.update_state(target_track_peaks, pred_track_full)
        auc_name = "_".join([track_name,
                             f"gt{pct_threshold}PredsFull", auc.name])
        results[auc_name] = auc.result().numpy()
        auprc_name = "_".join([track_name,
                               f"gt{pct_threshold}PredsFull", auprc.name])
        results[auprc_name] = auprc.result().numpy()
    return results


def evaluate_peak_predictions(targets, preds, track_names,
                              auc_implementation=None,
                              clear_session=True):
    """
    given (n_bins, n_tracks) arrays of targets and preds,
    whose 1st dimension corresponds to tracks specified
    in track_names, convert to binary peak predictions and 
    evaluate using classification metrics

    using sklearn implementation for auc, auprc as it seems
    to be much quicker and probably also more accurate.
    """
    # TODO: add spearman correlation
    results = {}
    print("Evaluating peak prediction using classification metrics", flush=True)
    classification_metrics = {
        "precision": tf.keras.metrics.Precision(),
        "recall": tf.keras.metrics.Recall(),
        }
    if tf.is_tensor(targets):
        print("Converting targets to np array")
        targets = targets.numpy()
    if tf.is_tensor(preds):
        print("Converting preds to np array")
        preds = preds.numpy()

    peak_eval_settings = [{"targets_peak_frac": 0.05,
                           "preds_peak_frac": 0.01,
                           "metrics": ["precision"],
                           "name": "gt5Preds1",
                           "preds_rescale": None},
                          {"targets_peak_frac": 0.01,
                           "preds_peak_frac": 0.05,
                           "metrics": ["recall"],
                           "name": "gt1Preds5",
                           "preds_rescale": None}]
    for sett in peak_eval_settings:
        targets_threshold = np.percentile(targets,
            100 - 100*sett["targets_peak_frac"], axis=0)
        preds_threshold = np.percentile(preds,
            100 - 100*sett["preds_peak_frac"], axis=0)

        # binarised targets and preds.
        peak_targets = (targets > targets_threshold).astype(int)
        peak_preds = (preds > preds_threshold).astype(int)

        # print(f"binarised vals")
        # print(f"Shapes: peak_preds {peak_preds.shape} gt_y {peak_targets.shape}")
        # print(f"nans: {np.sum(np.argwhere(~np.isfinite(peak_preds)))}"
        #       f" {np.sum(np.argwhere(~np.isfinite(peak_targets)))}"
        #       f" {np.sum(np.argwhere(~np.isfinite(preds)))}")

        for (track_i, track_name) in enumerate(track_names):
            target_track_peaks = peak_targets[:, track_i]
            pred_track_peaks = peak_preds[:, track_i]
            for m in sett["metrics"]:
                metric = classification_metrics[m]
                metric.reset_states()
                metric.update_state(target_track_peaks,
                                    pred_track_peaks)
                metric_name = "_".join([track_name, sett["name"],
                                        metric.name])
                results[metric_name] = metric.result().numpy()

    if auc_implementation is None:
        if SKLEARN_IS_AVAILABLE:
            print("defaulting to sklearn auc implementation")
            auc_implementation = "sklearn"
        else:
            print("defaulting to keras auc implementation")
            auc_implementation = "keras"

    if auc_implementation == "sklearn":
        assert SKLEARN_IS_AVAILABLE, "sklearn not available"
        results.update(_sklearn_auc(targets, preds, track_names,
                                    peak_threshold=0.01))
    elif auc_implementation == "keras":
        results.update(_keras_auc(targets, preds, track_names,
                                  peak_threshold=0.01))
    else:
        raise ValueError("auc_implementation must be either sklearn or keras")

    if clear_session:
        tf.keras.backend.clear_session()
    return results

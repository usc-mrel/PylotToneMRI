import joblib
import numpy as np
from scipy.signal import firwin, convolve
from scipy.interpolate import pchip_interpolate
import warnings

def pred_scan(rocket_pipeline, scan, force_navpred=False):
    ''' Predict navigators in the input scan using a pre-trained ROCKET classifier, given a set of sources.
    A label of 0 indicates non-navigator, 1 indicates respiratory navigator, and 2 indicates cardiac navigator.
    Parameters
    ----------
    rocket_pipeline : sklearn.pipeline.Pipeline
        Pre-trained ROCKET classifier pipeline.
    scan : np.ndarray
        Input scan sources, shape (n_sources, n_samples)
    force_navpred : bool
        If True, forces the function to return a respiratory and cardiac navigator even if the classifier does not predict any.
    Returns
    -------
    y_pred_ : np.ndarray
        Predicted labels for each source, shape (n_sources,).
    confs_ : np.ndarray
        Confidence scores for each source, shape (n_sources, n_classes).
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Workaround, ignore sklearn tag warnings.
        confs_ = rocket_pipeline.decision_function(scan[:,None,:])

    y_pred_ = np.argmax(confs_, axis=1)

    # Resolve multiple positive predictions
    if np.sum(y_pred_ == 1) > 1:
        resp_preds = (y_pred_ == 1).nonzero()[0]
        conf_r_ = confs_[resp_preds, 1]
        top_resp_idx = resp_preds[np.argmax(conf_r_)]
        other_idxs = np.setdiff1d(resp_preds, np.array([top_resp_idx]))
        y_pred_[other_idxs] = 0
        warnings.warn(f"Multiple respiratory predictions found at indices {resp_preds}, keeping index {top_resp_idx} only.")
    if np.sum(y_pred_ == 2) > 1:
        card_preds = (y_pred_ == 2).nonzero()[0]
        conf_r_ = confs_[card_preds, 2]
        top_card_idx = card_preds[np.argmax(conf_r_)]
        other_idxs = np.setdiff1d(card_preds, np.array([top_card_idx]))
        y_pred_[other_idxs] = 0
        warnings.warn(f"Multiple cardiac predictions found at indices {card_preds}, keeping index {top_card_idx} only.")
    if force_navpred:
        if np.sum(y_pred_ == 1) == 0:
            y_pred_[np.argmax(confs_[:,1])] = 1
            warnings.warn("No respiratory prediction was found, but force_navpred is True. Forcing the highest confidence prediction as respiratory.")
        if np.sum(y_pred_ == 2) == 0:
            y_pred_[np.argmax(confs_[:,2])] = 2
            warnings.warn("No cardiac prediction was found, but force_navpred is True. Forcing the highest confidence prediction as cardiac.")
    return y_pred_, confs_


def pick_navigators_from_sources(sources, time_vec, classifier_path='rocket_pipeline.pkl', force_navpred=False):
    ''' Pick respiratory and cardiac navigators from input sources using a pre-trained ROCKET classifier.
    Parameters
    ----------
    sources : np.ndarray
        Input sources, shape (n_sources, n_samples)
    time_vec : np.ndarray
        Time vector corresponding to the sources, unit is seconds, shape (n_samples,)
    classifier_path : str
        Path to the pre-trained ROCKET classifier. Must be compatible with joblib.load().
    force_navpred : bool
        If True, forces the function to return a respiratory and cardiac navigator even if the classifier does not predict any.
    Returns
    -------
    resp_idx : int
        Index of the respiratory navigator source in the input sources.
    card_idx : int
        Index of the cardiac navigator source in the input sources.
    confs : np.ndarray
        Confidence scores for each source, shape (n_sources, n_classes).
    '''
    rocket_pipeline = joblib.load(classifier_path)
    n_samp = sources.shape[1]
    dt_samp = time_vec[1] - time_vec[0]

    h_denoise = firwin(2*(n_samp//8)-1, [0.1, 6], fs=1/dt_samp, window=('tukey', 1), pass_zero=False)
    sources_filt = convolve(sources, h_denoise[None, :], mode='same')

    dt_new = 10e-3  # 10 ms
    n_samp_new = int(np.ceil(n_samp * dt_samp / dt_new))
    time_new = np.arange(0, n_samp_new)*dt_new
    sources_resampled = pchip_interpolate(time_vec, sources_filt, time_new, axis=1)
    sources_resampled -= np.mean(sources_resampled, axis=1, keepdims=True)
    sources_resampled /= np.std(sources_resampled, axis=1, keepdims=True)

    y_pred, confs = pred_scan(rocket_pipeline, sources_resampled, force_navpred=force_navpred)
    resp_idx = np.where(y_pred == 1)[0]
    card_idx = np.where(y_pred == 2)[0]
    return resp_idx, card_idx, confs

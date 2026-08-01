import math
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp
from numpy.fft import fft, ifft
from scipy.linalg import lstsq
from scipy.signal import find_peaks, peak_widths, savgol_filter, firwin, filtfilt
from scipy.signal.windows import tukey
from scipy.sparse.linalg import svds
import logging

from .signal import (
    angle_dependant_filtering,
    apply_filter_freq,
    cfftn,
    designbp_tukeyfilt_freq,
    designlp_tukeyfilt_freq,
    find_freq_qifft,
)
from .sobi import sobi
from .trajectory import calc_fovshift_phase

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import ismrmrd


def est_dtft(t, data, deltaf, window):
    ''' est_dtft MMSE sine amplitude estimate by DTFT sum given freq deltaf.
    Also subtracts the estimated windowed sine from the data.
     Inputs:
       t:      Time axis along readout [s].
       data:   (Nx x Nline x Nch) Time data to estimate peak amplitudes. 
       deltaf: Frequency where the peak occurs [Hz].
       window: Window function to multiply sine model before subtraction.
     Outputs:
       clean:  (Nx x Nline x Nch) Model subtracted data
       x_fit:  (Nline x Nch) Estimated complex amplitudes.'''

    dt = t[1]-t[0] # [s]
    x_fit = dtft_sum(data, dt, deltaf)
    s = (np.exp(-1j*2*np.pi*(deltaf[None,:])*t[:,None])*window[:,None])[:,:,None]
    clean = data - s*x_fit

    return (clean, x_fit)

def dtft_sum(data, dt, deltaf):
    Nsamp = data.shape[0]
    w0 = 2*deltaf*dt*np.pi # Normalized frequency [-pi, +pi]
    xsum = (np.sum(data*np.exp(+1j*w0[None,:,None]*np.arange(0, Nsamp)[:,None,None]),axis=0, keepdims=True)/Nsamp)
    return xsum


def extract_raw_pt(ksp_measured: np.ndarray, kx: np.ndarray, ky: np.ndarray,
                    n_unique_angles: int, acq: "ismrmrd.Acquisition", 
                    f_diff: float, df: float, dt: float, method: Literal['sine', 'wPCA'] = 'sine', 
                    freq_correction: bool = True, return_complex: bool = False) -> tuple[np.ndarray, np.ndarray]:
    
    n_acq = ksp_measured.shape[1]
    # ================================
    # Demodulate any shifts
    # ================================
    phase_mod_rads = calc_fovshift_phase(kx, ky, acq)
    phase_mod_rads = [phase_mod_rads[:,ii%n_unique_angles] for ii in range(n_acq)]
    phase_mod_rads = np.array(phase_mod_rads)[:, :].transpose()[:,:,None]

    # Apply the negative of the phase
    ksp_measured_ = ksp_measured*phase_mod_rads

    if freq_correction:
        fcorrmin = find_freq_qifft(ksp_measured_[:,:,:], df, f_diff, 3e3, 4, (2))
    else:
        fcorrmin = 0

    ksp_window = np.ones(ksp_measured_.shape[0])
    ksp_measured_ = ksp_measured_*ksp_window[:,None,None]

    time_acq = np.arange(0, ksp_measured_.shape[0])*dt

    if method == 'sine':
        ksp_ptsubbed_, pt_sig_fit = est_dtft(time_acq, ksp_measured_, np.array([f_diff])-fcorrmin, ksp_window)

    elif method == 'wPCA':
        if freq_correction:
            w_corr = np.exp(-2j*np.pi*np.arange(ksp_measured_.shape[0])[:,None]*dt*fcorrmin[None,:])[:,:,None]
        else:
            w_corr = 1
        
        X = np.reshape(ksp_measured_*w_corr, (ksp_measured_.shape[0], -1))
        b, _,_ = svds(X, k=1)
        
        B = np.reshape(b @ np.conj(b.T) @ X, (ksp_measured_.shape[0], n_acq, -1))*np.conj(w_corr)

        ksp_ptsubbed_ = ksp_measured_ - B
        # TODO: This is so weird. this function looks like it should return the same thing as dtft_sum, but it doesn't. It uses dtft_sum internally.
        # But jitter is smaller using this function. Maybe return back to this later.
        _, pt_sig_fit = est_dtft(time_acq, ksp_measured_, np.array([f_diff])-fcorrmin, ksp_window)
        # pt_sig_fit = (np.conj(X.T) @ b).reshape(n_acq, -1)

        # pt_sig_fit = pt.dtft_sum(ksp_ptsubbed_, dt, np.array([f_diff])-fcorrmin).squeeze()
        # plt.figure()
        # plt.subplot(121)
        # plt.plot(np.abs(b))
        # plt.title('Carrier estimated')
        # plt.subplot(212)
        # plt.plot(np.abs(pt_sig_fit))
        # plt.title('Motion')
    
    ksp_ptsubbed = ksp_ptsubbed_*np.conj(phase_mod_rads)

    if return_complex:
        pt_sig = np.squeeze(pt_sig_fit)
    else:
        pt_sig_fit = np.abs(pt_sig_fit)
        pt_sig = np.squeeze(pt_sig_fit - np.mean(pt_sig_fit, axis=1, keepdims=True))
        pt_sig = angle_dependant_filtering(pt_sig, n_unique_angles)
    return pt_sig, ksp_ptsubbed

def sniffer_sub(b: npt.NDArray, A: npt.NDArray):
    Npe = A.shape[0]
    filt = np.hstack((0, tukey(Npe-1, 0.6)))[:,None]
    
    A_f = np.real(ifft(fft(A)*filt/Npe))
    b_f = np.real(ifft(fft(b)*filt/Npe))
    # x = A_f\b_f LSQ
    x,_,_,_ = lstsq(A_f, b_f)
    clean = b - A.dot(x)

    return clean - np.mean(clean)

def plot_multich_comparison(tt: npt.NDArray[np.float64], sigs: tuple[npt.NDArray[np.float64], ...], 
                            titles: npt.ArrayLike, labels: tuple[str, ...]):
    # Plotting fcn
    n_ch = sigs[0].shape[1]
    n_sigs = len(sigs)
    if n_ch < 3:
        nc = 1
    else:
        nc = 2
    nr = math.ceil(n_ch/nc)
    ff, axs = plt.subplots(nr, nc, sharex=True)
    for ii in range(n_ch):
        xi = np.unravel_index(ii, (nr, nc))
        ax_ = axs[xi[0], xi[1]]

        for si in range(n_sigs):
            ax_.plot(tt, sigs[si][:,ii], label=labels[si])

        ax_.set_title(titles[ii])
        ax_.set_xlabel('Time [s]')

        if ii == 0:
            ax_.legend()

    # If number of chs is odd, last axes is empty, so remove
    if n_ch%2 == 1:
        axs[-1, -1].remove()


def pickcoilsbycorr(insig, start_ch, corr_th):
    Nch = insig.shape[1]
    C = np.corrcoef(insig, rowvar=False)

    # Automatic start_ch selector
    if start_ch == -1:
        C_ = np.copy(C)
        C_[np.abs(C_) < corr_th] = 0
        s = np.sum(np.abs(C_), axis=0)-1.0
        start_ch = np.argmax(s)

    accept_list = [start_ch]
    sign_list = [1]
    corrs = [1]

    for ii in range(Nch):
        if ii > start_ch:
            if abs(C[start_ch, ii]) > corr_th:
                accept_list.append(ii)
                sign_list.append(np.sign(C[start_ch, ii]))
                corrs.append(abs(C[start_ch, ii]))
        elif ii < start_ch:
            if abs(C[ii, start_ch]) > corr_th:
                accept_list.append(ii)
                sign_list.append(np.sign(C[ii, start_ch]))
                corrs.append(abs(C[ii, start_ch]))

    return accept_list, sign_list, corrs

def check_waveform_polarity(waveform: npt.NDArray[np.float64], prominence: float=0.5, method:Literal['std', 'width']='std') -> int:
    '''Check the polarity of the waveform and return the sign.
    The logic is, peaks looking up should be narrower than the bottom side for better triggering.
    
    Parameters:
    ----------
    waveform (np.array): Waveform to check.
    prominence (float): Prominence threshold for peak detection.
    method (str): Method to use for checking the polarity. 'std' for standard deviation of peak distances, 'width' for peak widths.

    Returns:
    ----------
    wf_sign (int): Sign of the waveform. 1 for positive, -1 for negative.
    '''
    waveform_ = waveform.copy()
    waveform_ -= np.percentile(waveform_, 5)
    waveform_ = waveform_/np.percentile(waveform_, 99)
    p1, d1 = find_peaks(waveform_, prominence=prominence)
    w1,_,_,_ = peak_widths(waveform_, p1)

    waveform_ = -1*waveform.copy()
    waveform_ -= np.percentile(waveform_, 5)
    waveform_ = waveform_/np.percentile(waveform_, 99)

    p2, d2 = find_peaks(waveform_, prominence=prominence)
    w2,_,_,_ = peak_widths(waveform_, p2)

    wf_sign = 1
    if method == 'std':
        if np.std(np.diff(p1)) > np.std(np.diff(p2)):
            wf_sign = -1
    elif method == 'width':
        if np.sum(w1) > np.sum(w2):
            wf_sign = -1

    return wf_sign

def extract_pilottone_navs(pt_sig, f_samp: float, params: dict):
    '''Extract the respiratory and cardiac pilot tone signals from the given PT signal.
    Parameters:
    ----------
    pt_sig (np.array): Pilot tone signal.
    f_samp (float): Sampling frequency of the PT signal.
    params (dict): Dictionary containing the parameters for the extraction.

    Returns:
    ----------
    pt_respiratory (np.array): Extracted respiratory pilot tone signal.
    pt_cardiac (np.array): Extracted cardiac pilot tone signal.
    '''
    n_pt_samp = pt_sig.shape[0]
    
    h_cardiac = firwin(2 * (n_pt_samp // 8) - 1, [params['cardiac']['freq_start'], params['cardiac']['freq_stop']], fs=f_samp, window=("tukey", 1), pass_zero=False)
    h_respiratory = firwin(2 * (n_pt_samp // 8) - 1, [params['respiratory']['freq_start'], params['respiratory']['freq_stop']], fs=f_samp, window=("tukey", 1), pass_zero=False)
    
    # Estimate actual length of the FIR filters.
    eps = 1e-9
    firlen_cardiac = np.sum(h_cardiac > eps)
    firlen_respiratory = np.sum(h_respiratory > eps)

    s_sobi, Asobi, Bsobi = sobi(pt_sig.T, num_lags=params['num_lags'])

    # Detect which channels are cardiac and respiratory navigators.
    r_idx2, r_stds2 = pick_source_bypeak(s_sobi.T, f_samp, fmask_low=0.2, fmask_high=0.6)
    c_idx2, c_stds2 = pick_source_bypeak(s_sobi.T, f_samp)

    logger.info(f"Picked respiratory source indices: {r_idx2}, stds: {r_stds2}")
    logger.info(f"Picked cardiac source indices: {c_idx2}, stds: {c_stds2}")
    pt_cardiac = filtfilt(h_cardiac, [1], s_sobi[c_idx2[np.argmin(c_stds2)], :], axis=0, method="gust", irlen=firlen_cardiac)
    pt_respiratory = filtfilt(h_respiratory, [1], s_sobi[r_idx2[0], :], axis=0, method="gust", irlen=firlen_respiratory)

    # Check and correct for the sign
    pt_cardiac = check_waveform_polarity(pt_cardiac[40:], method='width')*pt_cardiac

    return pt_respiratory, pt_cardiac

def extract_pilottone_navs_old(pt_sig, f_samp: float, params: dict):
    ''' This is the old method, replaced by a more robust and automated method, kept for reference.
    Extract the respiratory and cardiac pilot tone signals from the given PT signal.
    Parameters:
    ----------
    pt_sig (np.array): Pilot tone signal.
    f_samp (float): Sampling frequency of the PT signal.
    params (dict): Dictionary containing the parameters for the extraction.

    Returns:
    ----------
    pt_respiratory (np.array): Extracted respiratory pilot tone signal.
    pt_cardiac (np.array): Extracted cardiac pilot tone signal.
    '''
    n_pt_samp = pt_sig.shape[0]
    n_ch = pt_sig.shape[1]
    dt_pt = 1/f_samp
    time_pt = np.arange(n_pt_samp)*dt_pt
    
    # ================================================================
    # Denoising step
    # ================================================================ 
    
    pt_denoised = savgol_filter(pt_sig, params['golay_filter_len'], 3, axis=0)
    pt_denoised = pt_denoised - np.mean(pt_denoised, axis=0)

    if params['debug']['show_plots'] is True:
        plot_multich_comparison(time_pt, pt_sig, [' ']*n_ch, ['Original', 'SG filtered'])


    # ================================================================
    # Filter out higher than resp frequency ~1 Hz
    # ================================================================ 
    # df = f_samp/n_pt_samp/2
    # f_filt = np.arange(0, f_samp, df) - (f_samp - (n_pt_samp % 2)*df)/2 # Handles both even and odd length signals.

    if params['respiratory']['freq_start'] is None:
        filt_bp_resp = designlp_tukeyfilt_freq(params['respiratory']['freq_stop'], f_samp, n_pt_samp)
    else:
        filt_bp_resp = designbp_tukeyfilt_freq(params['respiratory']['freq_start'], params['respiratory']['freq_stop'], f_samp, n_pt_samp)

    pt_respiratory_freqs = apply_filter_freq(pt_denoised, filt_bp_resp, 'symmetric')

    if params['debug']['show_plots'] is True:
        plot_multich_comparison(time_pt, pt_respiratory_freqs, [' ']*n_ch, ['Original', 'respiratory filtered'])

    
    # ================================================================
    # Reject channels that have low correlation
    # ================================================================
    (accept_list, sign_list, corrs) = pickcoilsbycorr(pt_respiratory_freqs, params['respiratory']['corr_init_ch'], params['respiratory']['corr_threshold'])
    accept_list = np.sort(accept_list)
    print(f'Number of channels selected for respiratory PT: {len(accept_list)}')

    if params['respiratory']['separation_method'] == 'pca':
        # ================================================================
        # Apply PCA along coils to extract common signal (hopefuly resp)
        # ================================================================ 
        U, S, _ = svds(pt_respiratory_freqs[:,accept_list], k=1)

        # ================================================================
        # Separate a single respiratory source
        # ================================================================
        pt_respiratory = U*S
        pt_respiratory = pt_respiratory[:,0]

    elif params['respiratory']['separation_method'] == 'sobi':
        pt_respiratory, _, _ = sobi(pt_respiratory_freqs[:,accept_list].T)
        pt_respiratory = pt_respiratory[0,:]

    filt_bp_cardiac = designbp_tukeyfilt_freq(params['cardiac']['freq_start'], params['cardiac']['freq_stop'], f_samp, n_pt_samp)

    pt_cardiac_freqs = apply_filter_freq(pt_denoised, filt_bp_cardiac, 'symmetric')

    # Separate a single cardiac source
    # Correlation based channel selection
    # This is a semi automated fix for the case when a variety of SNR is
    # provided, corr_th needs to be adjusted. So, we start from high corr, and
    # loop until we have at least 2 channels with cardiac. My observation is,
    # if we can't find at least 2 channels, signal is too noisy to use anyways,
    # so we fail to extract cardiac PT.
    corr_threshold_cardiac = params['cardiac']['corr_threshold']
    while corr_threshold_cardiac >= min(0.5, corr_threshold_cardiac-0.05):
        [accept_list_cardiac, signList, corrChannels] = pickcoilsbycorr(pt_cardiac_freqs, params['cardiac']['corr_init_ch'], corr_threshold_cardiac)
        if len(accept_list_cardiac) < 2:
            corr_threshold_cardiac -= 0.05
        else:
            break

    if len(accept_list_cardiac) == 1:
        print('Could not find more channels with cardiac PT. Extraction is possibly failed.')

    print(f'Number of channels selected for cardiac PT: {len(accept_list_cardiac)}')
    if params['cardiac']['separation_method'] == 'pca':
        U, S, _ = svds(pt_cardiac_freqs[:,accept_list_cardiac], k=1)
        pt_cardiac = U*S
        pt_cardiac = pt_cardiac[:,0]
    elif params['cardiac']['separation_method'] == 'sobi':
        pt_cardiac, _, Vcard = sobi(pt_cardiac_freqs[:,accept_list_cardiac].T, num_lags=params['cardiac']['num_lags'])
        from .signal import cfftn
        # Determine which channel is the cardiac by looking at the frequency content
        df = f_samp/n_pt_samp
        faxis = np.arange(0, f_samp, df) - (f_samp - (n_pt_samp % 2)*df)/2
        f_mask = (faxis > 0.66) & (faxis < 3) # Assumes between 40 bpm to 180 bpm
        ptc_freq_max = np.max(np.abs(cfftn(pt_cardiac.T, axes=(0,))[f_mask,:]), axis=0)
        card_idx = np.argmax(ptc_freq_max)
        pt_cardiac = pt_cardiac[card_idx,:]

    # Normalize navs before returning.
    # Here, I am using prctile instead of the max to avoid weird spikes.
    if not params['debug']['no_normalize']:
        pt_respiratory -= np.percentile(pt_respiratory, 5)
        pt_respiratory /= np.percentile(pt_respiratory, 99)

        # Check if the waveform is flipped and flip if necessary.
        # Logic is, peaks looking up should be narrower than the bottom side for better triggering.
        ptc_sign = check_waveform_polarity(pt_cardiac[40:], prominence=0.5)
        pt_cardiac = ptc_sign*pt_cardiac
        
        # Shift the base and normalize again to make it mostly 0 to 1
        pt_cardiac -= np.percentile(pt_cardiac, 5)
        pt_cardiac = pt_cardiac/np.percentile(pt_cardiac, 99)

    return pt_respiratory, pt_cardiac

def calibrate_pt(pt_sig, f_samp: float, params: dict):
    '''Extract the respiratory and cardiac pilot tone signals from the given PT signal.
    Parameters:
    ----------
    pt_sig (np.array): Pilot tone signal.
    f_samp (float): Sampling frequency of the PT signal.
    params (dict): Dictionary containing the parameters for the extraction.

    Returns:
    ----------
    pt_respiratory (np.array): Extracted respiratory pilot tone signal.
    pt_cardiac (np.array): Extracted cardiac pilot tone signal.
    '''
    n_pt_samp = pt_sig.shape[0]
    n_ch = pt_sig.shape[1]
    dt_pt = 1/f_samp
    time_pt = np.arange(n_pt_samp)*dt_pt
    
    # ================================================================
    # Denoising step
    # ================================================================ 

    from scipy.signal import savgol_filter
    
    pt_denoised = savgol_filter(pt_sig, params['golay_filter_len'], 3, axis=0)
    pt_denoised = pt_denoised - np.mean(pt_denoised, axis=0)

    if params['debug']['show_plots'] is True:
        plot_multich_comparison(time_pt, pt_sig, pt_denoised, [' ']*n_ch, ['Original', 'SG filtered'])


    # ================================================================
    # Filter out higher than resp frequency ~1 Hz
    # ================================================================ 
    # df = f_samp/n_pt_samp/2
    # f_filt = np.arange(0, f_samp, df) - (f_samp - (n_pt_samp % 2)*df)/2 # Handles both even and odd length signals.

    if params['respiratory']['freq_start'] is None:
        filt_bp_resp = designlp_tukeyfilt_freq(params['respiratory']['freq_stop'], f_samp, n_pt_samp)
    else:
        filt_bp_resp = designbp_tukeyfilt_freq(params['respiratory']['freq_start'], params['respiratory']['freq_stop'], f_samp, n_pt_samp)

    pt_respiratory_freqs = apply_filter_freq(pt_denoised, filt_bp_resp, 'symmetric')

    if params['debug']['show_plots'] is True:
        plot_multich_comparison(time_pt, pt_denoised, pt_respiratory_freqs, [' ']*n_ch, ['Original', 'respiratory filtered'])

    
    # ================================================================
    # Reject channels that have low correlation
    # ================================================================
    (accept_list_resp, sign_list, corrs) = pickcoilsbycorr(pt_respiratory_freqs, params['respiratory']['corr_init_ch'], params['respiratory']['corr_threshold'])
    accept_list_resp = np.sort(accept_list_resp)
    print(f'Number of channels selected for respiratory PT: {len(accept_list_resp)}')

    if params['respiratory']['separation_method'] == 'pca':
        # ================================================================
        # Apply PCA along coils to extract common signal (hopefuly resp)
        # ================================================================ 
        Uresp, S, Vresp = svds(pt_respiratory_freqs[:,accept_list_resp], k=1)

        # ================================================================
        # Separate a single respiratory source
        # ================================================================
        pt_respiratory = Uresp
        pt_respiratory = pt_respiratory[:,0]

    elif params['respiratory']['separation_method'] == 'sobi':
        pt_respiratory, _, Vresp = sobi(pt_respiratory_freqs[:,accept_list_resp].T)
        pt_respiratory = pt_respiratory[0,:]

    filt_bp_cardiac = designbp_tukeyfilt_freq(params['cardiac']['freq_start'], params['cardiac']['freq_stop'], f_samp, n_pt_samp)

    pt_cardiac_freqs = apply_filter_freq(pt_denoised, filt_bp_cardiac, 'symmetric')

    # Separate a single cardiac source
    # Correlation based channel selection
    # This is a semi automated fix for the case when a variety of SNR is
    # provided, corr_th needs to be adjusted. So, we start from high corr, and
    # loop until we have at least 2 channels with cardiac. My observation is,
    # if we can't find at least 2 channels, signal is too noisy to use anyways,
    # so we fail to extract cardiac PT.
    corr_threshold_cardiac = params['cardiac']['corr_threshold']
    while corr_threshold_cardiac >= min(0.5, corr_threshold_cardiac-0.05):
        [accept_list_cardiac, signList, corrChannels] = pickcoilsbycorr(pt_cardiac_freqs, params['cardiac']['corr_init_ch'], corr_threshold_cardiac)
        if len(accept_list_cardiac) < 2:
            corr_threshold_cardiac -= 0.05
        else:
            break

    if len(accept_list_cardiac) == 1:
        print('Could not find more channels with cardiac PT. Extraction is possibly failed.')

    print(f'Number of channels selected for cardiac PT: {len(accept_list_cardiac)}')
    if params['cardiac']['separation_method'] == 'pca':
        Ucard, S, Vcard = svds(pt_cardiac_freqs[:,accept_list_cardiac], k=1)
        pt_cardiac = Ucard
        chs, stds = pick_cardiac_source(pt_cardiac, f_samp)
        pt_cardiac = pt_cardiac[chs[np.argmin(stds)],:]
    elif params['cardiac']['separation_method'] == 'sobi':
        pt_cardiac, _, Vcard = sobi(pt_cardiac_freqs[:,accept_list_cardiac].T, num_lags=params['cardiac']['num_lags'])
        chs, stds = pick_cardiac_source(pt_cardiac.T, f_samp)
        pt_cardiac = pt_cardiac[chs[np.argmin(stds)],:]

    # Normalize navs before returning.
    # Here, I am using prctile instead of the max to avoid weird spikes.
    if not params['debug']['no_normalize']:
        pt_respiratory -= np.percentile(pt_respiratory, 5)
        pt_respiratory /= np.percentile(pt_respiratory, 99)

        # Check if the waveform is flipped and flip if necessary.
        # Logic is, peaks looking up should be narrower than the bottom side for better triggering.
        ptc_sign = check_waveform_polarity(pt_cardiac[40:], prominence=0.5)
        pt_cardiac = ptc_sign*pt_cardiac
        
        # Shift the base and normalize again to make it mostly 0 to 1
        pt_cardiac -= np.percentile(pt_cardiac, 5)
        pt_cardiac = pt_cardiac/np.percentile(pt_cardiac, 99)

    return Vresp, accept_list_resp, pt_respiratory, Vcard, accept_list_cardiac, pt_cardiac

def apply_pt_calib(pt_sig, Vresp, accept_list_resp, Vcard, accept_list_cardiac, f_samp, params):
    '''Apply the calibration matrices to the PT signal.
    Parameters:
    ----------
    pt_sig (np.array): Pilot tone signal.
    Uresp (np.array): Respiratory calibration matrix.
    accept_list_resp (list): List of channels used for respiratory calibration.
    Ucard (np.array): Cardiac calibration matrix.
    accept_list_cardiac (list): List of channels used for cardiac calibration.

    Returns:
    ----------
    pt_respiratory (np.array): Extracted respiratory pilot tone signal.
    pt_cardiac (np.array): Extracted cardiac pilot tone signal.
    '''

    n_pt_samp = pt_sig.shape[0]
    n_ch = pt_sig.shape[1]
    dt_pt = 1/f_samp
    time_pt = np.arange(n_pt_samp)*dt_pt
    
    # ================================================================
    # Denoising step
    # ================================================================ 

    from scipy.signal import savgol_filter
    
    pt_denoised = savgol_filter(pt_sig, params['golay_filter_len'], 3, axis=0)
    pt_denoised = pt_denoised - np.mean(pt_denoised, axis=0)

    if params['debug']['show_plots'] is True:
        plot_multich_comparison(time_pt, pt_sig, pt_denoised, [' ']*n_ch, ['Original', 'SG filtered'])


    # ================================================================
    # Filter out higher than resp frequency ~1 Hz
    # ================================================================ 
    # df = f_samp/n_pt_samp/2
    # f_filt = np.arange(0, f_samp, df) - (f_samp - (n_pt_samp % 2)*df)/2 # Handles both even and odd length signals.

    if params['respiratory']['freq_start'] is None:
        filt_bp_resp = designlp_tukeyfilt_freq(params['respiratory']['freq_stop'], f_samp, n_pt_samp)
    else:
        filt_bp_resp = designbp_tukeyfilt_freq(params['respiratory']['freq_start'], params['respiratory']['freq_stop'], f_samp, n_pt_samp)

    pt_respiratory_freqs = apply_filter_freq(pt_denoised, filt_bp_resp, 'symmetric')

    pt_respiratory = pt_respiratory_freqs[:, accept_list_resp]@Vresp[:,0]

    filt_bp_cardiac = designbp_tukeyfilt_freq(params['cardiac']['freq_start'], params['cardiac']['freq_stop'], f_samp, n_pt_samp)
    pt_cardiac_freqs = apply_filter_freq(pt_denoised, filt_bp_cardiac, 'symmetric')

    pt_cardiac = pt_cardiac_freqs[:, accept_list_cardiac]@Vcard[:,0]


    return pt_respiratory, pt_cardiac

def process_cplx_pt(pt_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ''' Extract phase and magnitude from given complex PT signals. Processses the phase according to the method described in Supporting Information 1 of:
    Anand S, Lustig M. Beat Pilot Tone (BPT): Simultaneous MRI and RF motion sensing at arbitrary frequencies. Magnetic Resonance in Medicine. 2024;92(4):1768-1787. doi:10.1002/mrm.30150

    Parameters:
    ----------
    pt_raw (np.array): Complex PT signal of shape (n_samples, n_channels).
    
    Returns:
    ----------
    pt_raw_mag (np.array): Zero mean magnitude of the PT signal.
    pt_raw_ang (np.array): Phase of the PT signal.
    '''
    n_ch = pt_raw.shape[1]

    def Sr(r, n_ch):
        S_ = np.zeros((n_ch, n_ch))
        S_[:, r] = -1
        return S_
    
    def Xphr(X, r):
        return np.angle(X[..., r].conj()[:, None]*X)

    def Xph(X, n_ch):
        return np.concatenate([Xphr(X, r) for r in range(n_ch)], axis=1)

    P = np.zeros((n_ch*n_ch, n_ch))

    for ch_ in range(n_ch):
        P[ch_*n_ch:(ch_+1)*n_ch, :] = np.eye(n_ch) + Sr(ch_, n_ch)

    Pinv = np.linalg.pinv(P)

    Xph_ = Xph(pt_raw, n_ch)
    pt_raw_ang = (Pinv @ Xph_.T).T
    pt_raw_ang = np.squeeze(pt_raw_ang - np.mean(pt_raw_ang, axis=0, keepdims=True))

    pt_raw_mag = np.abs(pt_raw)
    pt_raw_mag = np.squeeze(pt_raw_mag - np.mean(pt_raw_mag, axis=0, keepdims=True))

    return pt_raw_mag, pt_raw_ang

def pick_cardiac_source(latent_vectors: np.ndarray, f_samp: float, fmask_low: float=0.66, fmask_high: float=3.0) -> tuple[np.ndarray, np.ndarray]:
    ''' Picks the cardiac source from the latent vectors using frequency content analysis. 
    Also computes the standard deviation of the envelope of the picked sources.

    Parameters:
    ----------
    latent_vectors (np.array): Latent vectors of shape (n_samples, n_sources).
    f_samp (float): Sampling frequency of the PT signal.
    fmask_low (float): Lower frequency bound for cardiac frequency mask [Hz].
    fmask_high (float): Upper frequency bound for cardiac frequency mask [Hz].

    Returns:
    ----------
    idxs (np.ndarray): Indices of picked cardiac sources.
    std_env (np.ndarray): Corresponding standard deviations of envelopes.
    '''

    n_pt_samp = latent_vectors.shape[0]
    df = f_samp/n_pt_samp
    faxis = np.linspace(0, f_samp, n_pt_samp) - (f_samp - (n_pt_samp % 2)*df)/2
    f_mask = (faxis > fmask_low) & (faxis < fmask_high) # By default, assumes between 40 bpm to 180 bpm
    ptc_freq_max = np.max(np.abs(cfftn(latent_vectors, axes=(0,))[f_mask,:]), axis=0)
    picked_by_spec = ptc_freq_max > np.max(ptc_freq_max)*0.9
    xenv, _ = sp.signal.envelope(latent_vectors[n_pt_samp//10:(-n_pt_samp//10), picked_by_spec], bp_in=(1,80), axis=0)
    std_env = np.std(xenv, axis=0)

    return np.where(picked_by_spec)[0], std_env

def pick_source_bypeak(latent_vectors: np.ndarray, f_samp: float, fmask_low: float=0.66, fmask_high: float=3.0, threshold=0.9) -> tuple[np.ndarray, np.ndarray]:
    ''' Similar to pick_cardiac_source, but with adjustable threshold.
    Picks the cardiac source from the latent vectors using frequency content analysis.
    Also computes the standard deviation of the envelope of the picked sources.
    Parameters:
    ----------
    latent_vectors (np.array): Latent vectors of shape (n_samples, n_sources).
    f_samp (float): Sampling frequency of the PT signal.
    fmask_low (float): Lower frequency bound for cardiac frequency mask [Hz].
    fmask_high (float): Upper frequency bound for cardiac frequency mask [Hz].
    threshold (float): Threshold for picking sources based on frequency content.
    Returns:
    ----------
    idxs (np.ndarray): Indices of picked cardiac sources.
    std_env (np.ndarray): Corresponding standard deviations of envelopes.
    '''
    
    n_pt_samp = latent_vectors.shape[0]
    df = f_samp/n_pt_samp
    faxis = np.linspace(0, f_samp, n_pt_samp) - (f_samp - (n_pt_samp % 2)*df)/2
    f_mask = (faxis > fmask_low) & (faxis < fmask_high) # By default, assumes between 40 bpm to 180 bpm
    ptc_freq_max = np.max(np.abs(cfftn(latent_vectors, axes=(0,))[f_mask,:]), axis=0)
    picked_by_spec = ptc_freq_max > np.max(ptc_freq_max)*threshold
    xenv, _ = sp.signal.envelope(latent_vectors[n_pt_samp//10:(-n_pt_samp//10), picked_by_spec], bp_in=(1,80), axis=0)
    std_env = np.std(xenv, axis=0)

    return np.where(picked_by_spec)[0], std_env

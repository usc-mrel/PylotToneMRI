import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy.fft import fft, ifft
from scipy.linalg import lstsq
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.signal.windows import tukey
from scipy.sparse.linalg import svds

from .signal import apply_filter_freq, designbp_tukeyfilt_freq, designlp_tukeyfilt_freq
from .sobi import sobi


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

    Nsamp = data.shape[0]
    dt = t[1]-t[0] # [s]
    w0 = 2*deltaf*dt*np.pi # Normalized frequency [-pi, +pi]
    
    x_fit = (np.sum(data*np.exp(+1j*w0[None,:,None]*np.arange(0, Nsamp)[:,None,None]),axis=0, keepdims=True)/Nsamp)

    s = (np.exp(-1j*2*np.pi*(deltaf[None,:])*t[:,None])*window[:,None])[:,:,None]
    clean = data - s*x_fit

    return (clean, x_fit)


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

def check_waveform_polarity(waveform: npt.NDArray[np.float64], prominence: float=0.5) -> int:
    '''Check the polarity of the waveform and return the sign.
    The logic is, peaks looking up should be narrower than the bottom side for better triggering.
    
    Parameters:
    ----------
    waveform (np.array): Waveform to check.
    prominence (float): Prominence threshold for peak detection.

    Returns:
    ----------
    wf_sign (int): Sign of the waveform. 1 for positive, -1 for negative.
    '''
    waveform_ = waveform.copy()
    waveform_ -= np.percentile(waveform_, 5)
    waveform_ = waveform_/np.percentile(waveform_, 99)
    p1, d1 = find_peaks(waveform_, prominence=prominence)
    w1,_,_,_ = peak_widths(waveform_, p1)

    waveform_ = -waveform_
    waveform_ -= np.percentile(waveform_, 5)
    waveform_ = waveform_/np.percentile(waveform_, 99)

    p2, d2 = find_peaks(-waveform, prominence=prominence)
    w2,_,_,_ = peak_widths(-waveform, p2)

    wf_sign = 1
    if np.sum(w1) > np.sum(w2):
        print('Cardiac waveform looks flipped. Flipping it..')
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
    n_ch = pt_sig.shape[1]
    dt_pt = 1/f_samp
    time_pt = np.arange(n_pt_samp)*dt_pt
    
    # ================================================================
    # Denoising step
    # ================================================================ 
    
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
    while corr_threshold_cardiac >= 0.5:
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
        pt_cardiac, _, _ = sobi(pt_cardiac_freqs[:,accept_list_cardiac].T)
        pt_cardiac = pt_cardiac[0,:]

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
    while corr_threshold_cardiac >= 0.5:
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
        pt_cardiac = pt_cardiac[:,0]
    elif params['cardiac']['separation_method'] == 'sobi':
        pt_cardiac, _, Vcard = sobi(pt_cardiac_freqs[:,accept_list_cardiac].T)
        pt_cardiac = pt_cardiac[0,:]

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

''' 
Contains signal processing functions.
Author: Bilal Tasdelen
'''

import numpy as np
import numpy.typing as npt
import pyfftw
import scipy as sp
from numpy.fft import fft, fftn, fftshift, ifft, ifftn, ifftshift
from numpy.polynomial import Polynomial
from scipy.signal.windows import tukey
from scipy.signal import firwin, convolve
from scipy.linalg import eigh


def cifft(data, axis):
    '''Centered IFFT.'''
    return ifftshift(ifft(fftshift(data, axis), None, axis), axis)

def cfft(data, axis):
    '''Centered FFT.'''
    return fftshift(fft(ifftshift(data, axis), None, axis), axis)

def cfftn(data, axes):
    '''Centered FFTN.'''
    return fftshift(fftn(ifftshift(data, axes=axes), None, axes=axes), axes=axes)

def cifftn(data, axes):
    '''Centered FFTN.'''
    return ifftshift(ifftn(fftshift(data, axes=axes), None, axes=axes), axes=axes)

def centered_crop(image, crop_size):
    """
    Crop the center of the image to the specified size.
    """
    center = np.array(image.shape) // 2
    start = center - np.array(crop_size) // 2
    end = start + np.array(crop_size)
    return image[tuple(slice(s, e) for s, e in zip(start, end))]

def rssq(data, axis):
    '''Root sum of squares along the given axis.'''
    return np.sqrt(np.sum(np.abs(data)**2, axis=axis))

def to_hybrid_kspace(indata):
    '''Centered ifft on first dimension. Does not do fftshift before ifft, as it treats data as time signal.'''
    return ifftshift(ifft(indata, None, axis=0), axes=0)

def from_hybrid_kspace(indata):
    '''Centered ifft on first dimension. Does not do fftshift before ifft, as it treats data as time signal.'''
    return fft(fftshift(indata, axes=0), None, axis=0)


def qint(ym1, y0, yp1):
    '''Quadratic interpolation.
        Parameters
        ----------
        ym1 : float or array
            First point.
        y0 : float or array
            Middle point.        
        yp1 : float or array
            Last point.
        
        Returns
        -------
        p : float or array
            Shift of peak (or dip) from y0
        y : float or array
            slope
        a : float or array
            bias
    '''
    p = (yp1 - ym1) / (2 * (2 * y0 - yp1 - ym1))
    y = y0 - 0.25 * (ym1 - yp1) * p
    a = 0.5 * (ym1 - 2 * y0 + yp1)
    return p, y, a

def find_freq_qifft(data, df, f_center, f_radius, os, ave_dim):
    Nsamp = data.shape[0]
    dfint = df / os
    data = pyfftw.byte_align(data.transpose(2,1,0))

    # start_time = time.time()
    fft = pyfftw.builders.ifft(data, n=Nsamp*os, axis=2, threads=64, planner_effort='FFTW_ESTIMATE')
    data_f = np.fft.ifftshift(fft(), axes=2)
    data_f = data_f.transpose((2, 1 ,0))
    # end_time = time.time()
    # print(end_time-start_time)
    # data_f = np.fft.ifftshift(pyfftw.interfaces.numpy_fft.ifft(data, Nsamp*os, axis=0), axes=0)
    # data_f = np.fft.ifftshift(np.fft.ifft(data, Nsamp * os, axis=0), axes=0)
    data_f = np.abs(data_f)

    if ave_dim is not None:
        data_fpk = np.mean(data_f, axis=ave_dim)

    f_axis = np.arange(-Nsamp * os / 2, Nsamp * os / 2) * dfint

    f_search_interval = (f_axis < (f_center + f_radius / 2)) & (f_axis > (f_center - f_radius / 2))

    if np.sum(f_search_interval) == 0:
        raise ValueError('Search frequency is outside of the imaging bandwidth. Check PT frequency.')

    data_fpk_srch = data_fpk[f_search_interval]
    f_axis_fpk_srch = f_axis[f_search_interval]

    Iinit = np.argmax(data_fpk_srch, axis=0)

    if np.any((Iinit == 0) | (Iinit == len(data_fpk_srch) - 1)):
        print(f'Peak is found at the edge index of {Iinit}.\nThis may mean the peak is outside of the given frequency range or there is no peak at all. Returning 0.')
        return 0

    finit = f_axis_fpk_srch[Iinit]
    if data_fpk_srch.ndim > 1: 
        Ipr = np.ravel_multi_index((Iinit-1, np.arange(data_fpk_srch.shape[1])), data_fpk_srch.shape)
        Icr = np.ravel_multi_index((Iinit, np.arange(data_fpk_srch.shape[1])), data_fpk_srch.shape)
        Inx = np.ravel_multi_index((Iinit+1, np.arange(data_fpk_srch.shape[1])), data_fpk_srch.shape)
    else:
        Ipr = Iinit-1
        Icr = Iinit
        Inx = Iinit+1

    p, _, _ = qint(data_fpk_srch.ravel()[Ipr], data_fpk_srch.ravel()[Icr], data_fpk_srch.ravel()[Inx])

    f_found = finit + p * dfint
    fcorrmin = f_center - f_found

    return fcorrmin

def designlp_tukeyfilt_freq(Fstop: float, Fs: float, Ns: int):
    '''Design frequency coefficients for a low-pass filter using Tukey window in frequency domain.
        Parameters
        ----------
        Fstop : float
            Stop frequency in Hz.
        Fs : float
            Sampling frequency in Hz.
        Ns : int
            Filter length.
        
        Returns
        -------
        filter: 1D array
            Filter coefficients in frequency domain.
    '''
    Ns = 2*Ns
    df = Fs/Ns
    n_pass = 2*round(Fstop/df)+1
    twin = tukey(n_pass, 0.3)
    return np.vstack((np.zeros((int((Ns+1-n_pass)/2),1)), twin[:,None], np.zeros((int((Ns-1-n_pass)/2),1))))

def designhp_tukeyfilt_freq(Fstart, Fs, Ns):
    '''Design frequency coefficients for a high-pass filter using inverted low-pass filter.
            Parameters
        ----------
        Fstart : float
            Start frequency in Hz.
        Fs : float
            Sampling frequency in Hz.
        Ns : int
            Filter length.
        
        Returns
        -------
        filter: 1D array
            Filter coefficients in frequency domain.
        '''
    return 1 - designlp_tukeyfilt_freq(Fstart, Fs, Ns)

def designbp_tukeyfilt_freq(Fstop1, Fstop2, Fs, Ns):
    '''Design frequency coefficients for a band-pass filter two low-pass filters.
        Parameters
        ----------
        Fstop1 : float
            Start frequency in Hz.
        Fstop2: float
            Stop frequency in Hz.
        Fs : float
            Sampling frequency in Hz.
        Ns : int
            Filter length.
        
        Returns
        -------
        filter: 1D array
            Filter coefficients in frequency domain.
    '''

    filtlp1 = designlp_tukeyfilt_freq(Fstop1, Fs, Ns)
    filtlp2 = designlp_tukeyfilt_freq(Fstop2, Fs, Ns)
    return filtlp2 - filtlp1

def apply_filter_freq(sig: npt.NDArray[np.float32], flt: npt.NDArray[np.complex64], pad_method: str):
    '''Filter the signal with the given frequency coefficients.
        Parameters
        ----------
        sig : NDArray
            2D Array that will be filtered in first dimension.
        flt : NDArray
            Filter coefficients in frequency domain.
        pad_method : str
            How will the signal be padded for linear convolution in frequency domain.
            'negflip': Will flip and negate a portion of the original signal for padding.
            'symmetric': Will flip the signal in time as padding.
        
        Returns
        -------
        sig_filt : NDArray
            Filtered signal.
    '''
    N = sig.shape[0]
    
    if pad_method == 'negflip':
        R = 0.1  # 10% of signal
        Nr = 800
        NR = min(round(N * R), Nr)  # At most 50 points
        x1 = 2 * sig[0, :] - np.flipud(sig[1:NR+1, :])  # maintain continuity in level and slope
        x2 = 2 * sig[-1, :] - np.flipud(sig[-NR-1:-1, :])
        sig_padded = np.vstack([x1, sig, x2])
        sig_filt = np.real(cifft(cfft(np.pad(sig_padded, ((N//2-NR, 0), (0, 0)), mode='constant'), axis=0) * flt[:, np.newaxis], axis=0))

    elif pad_method == 'symmetric':
        sig_padded = np.pad(sig, ((N//2, N//2), (0, 0)), mode='symmetric')
        if N % 2 != 0:
            sig_padded = np.pad(sig_padded, ((0, 1), (0, 0)), mode='constant')
        sig_filt = np.real(cifft(cfft(sig_padded, axis=0) * flt, axis=0))

    sig_filt = sig_filt[N//2:(N//2 + N), :]
    
    return sig_filt

def filter_freq(sig: np.ndarray, Fstop1, Fstop2, Fs: float, pad_method:str ='symmetric'):
    '''Filter the signal between Fstop1 and Fstop2 using Tukey window in frequency domain.
        Parameters
        ----------
        sig : NDArray
            [Nsamp x Nch] 2D Array that will be filtered in first dimension.
        Fstop1 : float
            Start frequency in Hz.
        Fstop2: float
            Stop frequency in Hz.
        Fs : float
            Sampling frequency in Hz.
        pad_method : str
            How will the signal be padded for linear convolution in frequency domain.
            'negflip': Will flip and negate a portion of the original signal for padding.
            'symmetric': Will flip the signal in time as padding.
        
        Returns
        -------
        sig_filt : NDArray
            Filtered signal.
    '''
    N = sig.shape[0]
    flt = designbp_tukeyfilt_freq(Fstop1, Fstop2, Fs, N)
    sig_filt = apply_filter_freq(sig, flt, pad_method)
    return sig_filt

def firwin_filt(signal_in: np.ndarray, Fstart: float, Fstop: float, f_samp: float, num_taps: int = 1299) -> np.ndarray:
    ''' Shortcut for designing and applying single band-pass firwin filter to multichannel data.

    Parameters
    ----------
    signal_in : NDArray
        [Nsamp x Nch] Multichannel input signal to be filtered.
    Fstart : float
        Start frequency of the band-pass filter.
    Fstop : float
        Stop frequency of the band-pass filter.
    f_samp : float
        Sampling frequency of the input signal.
    num_taps : int
        Number of taps in the FIR filter.

    Returns
    -------
    NDArray
        Filtered signal.
    '''
    h = firwin(num_taps, [Fstart, Fstop], fs=f_samp, window=('tukey', 1), pass_zero=False)
    pt_filtered = convolve(signal_in, h[:, None], mode='same')
    return pt_filtered

def angle_dependant_filtering(sig: npt.NDArray[np.float64], n_unique_angles: int, angle_step:float=222.4922, pdegree:int=9) -> npt.NDArray[np.float64]:
    n_acq, nc = sig.shape[0:2]
    angs = np.arange(n_unique_angles)*angle_step % 360
    angs_sorted = np.sort(angs)
    sort_idx = np.argsort(angs)
    angle_idx = np.arange(n_acq) % n_unique_angles
    angles = angs[angle_idx]
    I_angles = np.argsort(angles)
    Irev = np.argsort(I_angles)

    counts = np.bincount(angle_idx, minlength=n_unique_angles).astype(np.float64)
    valid = counts > 0
    sig_mean_over_reps = np.full((n_unique_angles, nc), np.nan, dtype=np.float64)
    for chi in range(nc):
        sums = np.bincount(angle_idx, weights=sig[:, chi], minlength=n_unique_angles).astype(np.float64)
        sig_mean_over_reps[valid, chi] = sums[valid] / counts[valid]
    sig_mean_over_reps = sig_mean_over_reps[sort_idx]

    sig_filtered = np.zeros(sig.shape)
    for chi in range(nc):
        x = angs_sorted[valid[sort_idx]]
        y = sig_mean_over_reps[valid[sort_idx], chi]
        p = Polynomial.fit(x, y, deg=min(pdegree, len(x) - 1))
        sig_filtered[:,chi] = (sig[I_angles, chi] - p(angles[I_angles]))[Irev]

    return sig_filtered

def xcorr_channels(measurements: np.ndarray, reference_channel: int = 0):
    """
    Compute cross-correlation between channels and a reference channel.

    Parameters:
    measurements (np.ndarray): [n_samples x n_channels] 2D array where each column represents a channel.
    reference_channel (int): Index of the reference channel.

    Returns:
    delays (np.ndarray): List of delays for each channel relative to the reference channel.
    max_corrs (np.ndarray): List of maximum correlation values for each channel.
    pearson_corrs (np.ndarray): List of Pearson correlation coefficients for each channel.
    """
    delays = []
    max_corrs = []
    pearson_corrs = []

    n_samples = measurements.shape[0]
    for ch in range(measurements.shape[1]):
        corrs = sp.signal.correlate(measurements[:, reference_channel], measurements[:, ch], mode='same')
        pearson_corrs.append(np.corrcoef(measurements[:, reference_channel], measurements[:, ch])[0,1])
        max_corr_pos = np.argmax(np.abs(corrs))
        
        max_corrs.append(corrs[max_corr_pos]/(np.std(measurements[:, reference_channel])*np.std(measurements[:, ch])*n_samples))

        delay = max_corr_pos - n_samples//2
        delays.append(delay)

    return np.array(delays), np.array(max_corrs), np.array(pearson_corrs)

def gram_schmidt(A:np.ndarray) -> np.ndarray:
    ''' Gram-Schmidt orthogonalization of the columns of A. Code adapted from: https://www.sfu.ca/~jtmulhol/py4math/linalg/np-gramschmidt/
    
    Parameters
    ----------
    A : np.ndarray 
    A set of linearly independent vectors stored as the columns of matrix A
    
    Returns
    -------
    Aorth: np.ndarray
    An orthongonal basis for the column space of A.
    '''
    # get the number of vectors.
    A = np.copy(A).astype(np.float64) # create a local instance of the array
    n = A.shape[1]
    for j in range(n):
        # For the vector in column j, find the perpendicular
        # of the projection onto the previous orthogonal vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        # If original vectors aren't lin indep then we can check for this:
        #
        if np.isclose(np.linalg.norm(A[:, j]), 0, rtol=1e-15, atol=1e-14, equal_nan=False):
            A[:, j] = np.zeros(A.shape[0])
        else:    
            A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A

def calc_gevd(signal: np.ndarray, interference: np.ndarray, ortho: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the virtual coils using GEVD from the signal and interference data.
    
    Parameters
    ----------
    signal : np.ndarray
        (n_s x n_ch) flattened signal data.
    interference : np.ndarray
        (n_int x n_ch) flattened interference data.
    ortho : bool, optional
        Whether to orthogonalize the GEVD eigenvectors using Gram-Schmidt. Default is False.
        
    Returns
    -------
    Vgevd : np.ndarray
        (n_ch x n_ch) matrix of GEVD eigenvectors.
    SIR : np.ndarray
        Signal-to-Interference Ratio for each channel.
    SNR : np.ndarray
        Signal-to-Noise Ratio for each channel.
    """
    
    # Calculate A and B matrices
    A = signal.conj().T@signal
    B = interference.conj().T@interference

    # Solve for generalized eigenvalues
    D, V = eigh(A, B)

    idx = np.argsort(D)[::-1]  # Sort eigenvalues in descending order
    Vgevd = V[:, idx]  # Sort eigenvectors accordingly
    if ortho:
        Vgevd = gram_schmidt(Vgevd)
    else:
        Vgevd /= np.linalg.vector_norm(Vgevd, axis=0, keepdims=True, ord=2)  # Normalize eigenvectors

    # Calculate SNR and SIR
    nc = Vgevd.shape[1]
    SNR = np.zeros(nc)
    SIR = np.zeros(nc)
    for c_i in range(nc):
        W = Vgevd[:, c_i]
        SNR[c_i] = abs((W.conj().T @ A @ W) / (W.conj().T @ W))
        SIR[c_i] = abs((W.conj().T @ A @ W) / (W.conj().T @ B @ W))
    
    return Vgevd, SIR, SNR

def calc_rovir(im: np.ndarray, signal_mask: np.ndarray, interference_mask: np.ndarray, ortho: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the ROVIR coils from the image and masks.
    
    Parameters
    ----------
    im : np.ndarray
        (n_s x n_ch) flattened image data.
    signal_mask : np.ndarray
        (n_s x 1) Mask for the signal.
    interference_mask : np.ndarray
        (n_s x 1) Mask for the interference.
    ortho : bool, optional
        Whether to orthogonalize the ROVIR eigenvectors using Gram-Schmidt. Default is False.
        
    Returns
    -------
    Vrvr : np.ndarray
        (n_ch x n_ch) matrix of ROVIR eigenvectors.
    im_rvr : np.ndarray
        (n_s x n_ch) matrix of the image transformed to the ROVIR basis.
    SIR : np.ndarray
        Signal-to-Interference Ratio for each channel.
    SNR : np.ndarray
        Signal-to-Noise Ratio for each channel.
    """
    
    im_signal = im[signal_mask, :]
    im_interference = im[interference_mask, :]

    Vrvr, SIR, SNR = calc_gevd(im_signal, im_interference, ortho=ortho)

    # Transform PT to the new basis
    im_rvr = im @ Vrvr
    
    return Vrvr, im_rvr, SIR, SNR

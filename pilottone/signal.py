''' 
Contains signal processing functions.
Author: Bilal Tasdelen
'''

from numpy.fft import ifft, ifftshift, fft, fftshift
import numpy as np
import numpy.typing as npt
from scipy.signal.windows import tukey

def cifft(data, axis):
    '''Centered IFFT.'''
    return ifftshift(ifft(fftshift(data, axis), None, axis), axis)

def cfft(data, axis):
    '''Centered FFT.'''
    return fftshift(fft(ifftshift(data, axis), None, axis), axis)

def to_hybrid_kspace(indata):
    '''Centered ifft on first dimension. Does not do fftshift before ifft, as it treats data as time signal.'''
    return ifftshift(ifft(indata, None, axis=0), axes=0)

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

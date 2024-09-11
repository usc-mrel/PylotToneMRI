from warnings import warn
import numpy as np
import numpy.typing as npt
from numpy.fft import ifft, ifftshift, fft, fftshift
from typing import Union
import math
import scipy as sp

import cupy as cp
import cupy.typing as cpt

def to_hybrid_kspace(indata):
    '''Centered ifft on first dimension. Does not do fftshift before ifft, as it treats data as time signal.'''
    return ifftshift(ifft(indata, None, axis=0), axes=0)

def autopick_sensing_coils(data, f_emi, bw_emi, bw_sig, f_samp, ratio_th=None, n_sensing=None):

    n_samp = data.shape[0]
    df = f_samp/n_samp
    freq_axis = np.arange(0, f_samp, df) - (f_samp - (n_samp % 2)*df)/2 # Handles both even and odd length signals.
    signal_mask = (freq_axis < bw_sig/2) & (freq_axis > -bw_sig/2)
    signal_region = to_hybrid_kspace(data[:, 0,:].squeeze())*signal_mask[:,None]

    emi_mask = (freq_axis < (f_emi+bw_emi/2)) & (freq_axis > (f_emi-bw_emi/2))
    emi_region = to_hybrid_kspace(data[:, 0,:].squeeze())*emi_mask[:,None]

    emi_energy = np.sum(np.abs(emi_region), axis=0)
    signal_energy = np.sum(np.abs(signal_region), axis=0)
    sratio = signal_energy/emi_energy

    Isig = np.argsort(signal_energy)
    Irat = np.argsort(sratio)

    sratio /= sratio[Isig[-1]] # Normalize with the largest Signal coil to reduce dep on PT amplitude.

    # plt.figure()
    # plt.plot(np.abs(to_hybrid_kspace(data[:,0,:].squeeze())))

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(emi_energy, label='emi_energy')
    # ax1.plot(signal_energy, label='signal_energy')
    # ax1.legend()
    # ax2.plot(sratio, 'green', label='ratio')
    # ax2.legend()
    # plt.show()
    # plt.figure()
    # plt.plot(np.abs(signal_region))
    # plt.show()

    # plt.figure()
    # plt.plot(np.abs(emi_region))
    # plt.show()

    # print(f'Coil with highest signal {coil_name[Isig[-1]]}.')

    # print(coil_name[Isig])

    # print(coil_name[Irat])

    # print(f'Coils with smallest ratio {coil_name[sratio < ratio_th]}')

    n_ch = data.shape[2]
    if n_sensing is not None and ratio_th is not None:
        warn('Both n_sensing and ratio_th are set. Using n_sensing.')
    if n_sensing is not None:
        sensing_coils = Irat[:n_sensing]
    elif ratio_th is not None:
        sensing_coils = np.nonzero(sratio < ratio_th)[0]
    else:
        raise ValueError('Either n_sensing or ratio_th must be set.')

    mri_coils = np.arange(n_ch)
    mri_coils = mri_coils[~np.isin(mri_coils, sensing_coils)]
    return mri_coils, sensing_coils



def get_noise_mtx(line_grp: Union[npt.NDArray[np.complex64], cpt.NDArray[cp.complex64]], dk: list[int]):
    """
    Creates the shifted noise matrix for a given line group and kernel sizes.

    Args:
        line_grp (numpy.ndarray): Line group data of shape (Nsamples, Nlines, Nchannels).
        dk (list or tuple): Kernel size [d_kx, d_ky].

    Returns:
        numpy.ndarray: Noise matrix of shape ((Nsamples * Nlines) x (Nchannels * (d_kx * 2 + 1) * (d_ky * 2 + 1))).
    """
    xp = cp.get_array_module(line_grp)
    d_kx = dk[0]
    d_ky = dk[1]

    n_ch = line_grp.shape[2]

    # noise_mat = np.zeros((line_grp.shape[0], line_grp.shape[1], n_ch*(d_kx*2+1)*(d_ky*2+1)), dtype=line_grp.dtype)
    noise_mat = []

    dfp = xp.pad(line_grp, ((d_kx, d_kx), (d_ky, d_ky), (0, 0)), mode='constant')
    if d_ky == 0:
        end_slc = None
    else:
        end_slc = d_ky
    ii = 0
    for col_shift in range(-d_kx, d_kx + 1):
        for lin_shift in range(-d_ky, d_ky + 1):
            dftmp = xp.roll(dfp, shift=(col_shift, lin_shift), axis=(0, 1))
            cropped = dftmp[d_kx:-d_kx, d_ky:end_slc, :]
            # noise_mat[:,:,(ii*n_ch):((ii+1)*n_ch)] = dftmp[d_kx:-d_kx, d_ky:end_slc, :]
            ii += 1

            noise_mat.append(cropped)

    noise_mat = xp.concatenate(noise_mat, axis=2)
    noise_mat = noise_mat.reshape(noise_mat.shape[0] * noise_mat.shape[1], -1)

    return noise_mat

def est_emi(signal_in: npt.NDArray[np.complex64], sniffer: npt.NDArray[np.complex64], line_grps: list[npt.NDArray], dk: list[int], w: npt.NDArray[np.float32]):

    Ncol, Nlin, Nc = sniffer.shape
    emi_hat = np.zeros((Ncol, Nlin), dtype=np.complex64)
    kern = [] # np.zeros((Nc * (dk[0] * 2 + 1) * (dk[1] * 2 + 1), Ngrp))

    for cwin, pe_rng in enumerate(line_grps):
        # pe_rng is the range of lines in the current group
        noise_mat = get_noise_mtx(sniffer[:, pe_rng, :], dk)

        # Select grouped lines and flatten for inversion
        init_mat_sub = np.reshape(signal_in[:, pe_rng], (Ncol * len(pe_rng), 1))

        ww = np.reshape(w[:, pe_rng], (Ncol * len(pe_rng), 1))

        # Solving the linear system
        kern_ ,_,_,_ = sp.linalg.lstsq(ww * noise_mat, ww * init_mat_sub, cond=None, check_finite=False)

        kern.append((pe_rng, kern_))
        # Put the solution back
        emi_hat[:, pe_rng] = np.reshape(cp.dot(noise_mat, kern_), (Ncol, len(pe_rng)))

    return emi_hat, kern

def est_emi_gpu(signal_in: npt.NDArray[np.complex64], sniffer: npt.NDArray[np.complex64], line_grps: list[npt.NDArray], dk: list[int], w: npt.NDArray[np.float32]):
        
    Ncol, Nlin, Nc = sniffer.shape
    # Ngrp = len(line_grps)
    emi_hat = np.zeros((Ncol, Nlin), dtype=np.complex64)
    kern = [] # np.zeros((Nc * (dk[0] * 2 + 1) * (dk[1] * 2 + 1), Ngrp))
    for cwin, pe_rng in enumerate(line_grps):
        # pe_rng is the range of lines in the current group
        noise_mat = get_noise_mtx(cp.asarray(sniffer[:, pe_rng, :]), dk)

        # Select grouped lines and flatten for inversion
        init_mat_sub = cp.reshape(cp.asarray(signal_in[:, pe_rng]), (Ncol * len(pe_rng), 1))

        ww = cp.reshape(cp.asarray(w[:, pe_rng]), (Ncol * len(pe_rng), 1))

        # Solving the linear system
        kern_ ,_,_,_ = cp.linalg.lstsq(ww * noise_mat, ww * init_mat_sub, rcond=None)

        kern.append((pe_rng, kern_.get()))
        # Put the solution back
        emi_hat[:, pe_rng] = cp.reshape(cp.dot(noise_mat, kern_), (Ncol, len(pe_rng))).get()

    return emi_hat, kern

def apply_editer(signal_in: npt.NDArray[np.complex64], sniffer: npt.NDArray[np.complex64], params, w) -> tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]]:
    max_lines = params['max_lines_per_group']
    nlin = signal_in.shape[1]
    if params['grouping_method'] == "uniform":
        Ngrp = math.ceil(nlin/max_lines)
        line_grps = []

        for grp_i in range(Ngrp):
            line_grps.append(np.arange(((grp_i)*max_lines), min(max_lines*(grp_i+1), nlin)))


    # emi_hat, kernels = est_emi(signal_in, sniffer, line_grps, params['dk'], w)
    with cp.cuda.Device(2):
        emi_hat, kernels = est_emi_gpu(signal_in, sniffer, line_grps, params['dk'], w)
    return emi_hat, kernels
    
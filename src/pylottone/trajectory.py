import ismrmrd
import numpy as np
import numpy.typing as npt
import pyfftw

def calc_fovshift_phase(kx: npt.NDArray, ky: npt.NDArray, acq: ismrmrd.Acquisition) -> npt.NDArray[np.complex64]:
    '''Calculate the phase demodulation due to the FOV shift in the GCS.

    Parameters:
    ----------
    kx (np.ndarray): 
        1D array of k-space points in the logical x coordinates.
    ky (np.ndarray): 
        2D array of k-space points in the logical y coordinates.
    acq (ismrmrd.Acquisition): 
        Acquisition object containing the phase and read directions, and position.

    Returns:
    ----------
    np.ndarray: 
        1D array of phase demodulation values in the GCS.
    '''

    gbar = 42.576e6

    gx = np.diff(np.concatenate((np.zeros((1, kx.shape[1]), dtype=kx.dtype), kx)), axis=0)/gbar # [T/m]
    gy = np.diff(np.concatenate((np.zeros((1, kx.shape[1]), dtype=kx.dtype), ky)), axis=0)/gbar # [T/m]
    g_nom = np.stack((gx, gy), axis=2)
    g_gcs = np.concatenate((g_nom, np.zeros((g_nom.shape[0], g_nom.shape[1], 1))), axis=2)

    r_GCS2RCS = np.array(  [[0,    1,   0],  # [PE]   [0 1 0] * [r]
                            [1,    0,   0],  # [RO] = [1 0 0] * [c]
                            [0,    0,   1]]) # [SL]   [0 0 1] * [s]

    r_GCS2PCS = np.array([np.array(acq.phase_dir), 
                        np.array(acq.read_dir), 
                        np.array(acq.slice_dir)])
    PCS_offset = np.array([1, 1, 1])*np.array(acq.position)*1e-3
    GCS_offset = r_GCS2PCS.dot(PCS_offset)
    # RCS_offset = r_GCS2RCS.dot(GCS_offset)
    g_rcs = np.dot(r_GCS2RCS, np.transpose(g_gcs, (1,2,0))).transpose((2,1,0))
    phase_mod_rads = np.exp(-1j*np.cumsum(2*np.pi*gbar*np.sum(g_rcs*GCS_offset, axis=2), axis=0)) # [rad]

    return phase_mod_rads.astype(np.complex64)

def remove_readout_os(ksp_ptsubbed: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    '''Remove 2x readout oversampling from the k-space data.'''
    
    n_samp = ksp_ptsubbed.shape[0]
    ksp_ptsubbed = pyfftw.byte_align(ksp_ptsubbed)

    keepOS = np.concatenate([np.arange(n_samp // 4), np.arange(n_samp * 3 // 4, n_samp)])
    ifft_ = pyfftw.builders.ifft(ksp_ptsubbed, n=n_samp, axis=0, threads=32, planner_effort='FFTW_ESTIMATE')
    ksp_ptsubbed = ifft_()

    fft_ = pyfftw.builders.fft(ksp_ptsubbed[keepOS, :, :], n=keepOS.shape[0], axis=0, threads=32, planner_effort='FFTW_ESTIMATE')
    ksp_ptsubbed = fft_()
    return ksp_ptsubbed
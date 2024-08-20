import numpy as np
from numpy.fft import ifft, ifftshift, fft, fftshift

def to_hybrid_kspace(indata):
    '''Centered ifft on first dimension. Does not do fftshift before ifft, as it treats data as time signal.'''
    return ifftshift(ifft(indata, None, axis=0), axes=0)

def autopick_sensing_coils(data, f_emi, bw_emi, bw_sig, f_samp, ratio_th):

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
    # Irat = np.argsort(sratio)

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
    sensing_coils = np.nonzero(sratio < ratio_th)[0]
    mri_coils = np.arange(n_ch)
    mri_coils = mri_coils[~np.isin(mri_coils, sensing_coils)]
    return mri_coils, sensing_coils
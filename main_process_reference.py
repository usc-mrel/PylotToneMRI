# %%
import ismrmrd
import rtoml
import os
from scipy.io import loadmat
from scipy.signal.windows import tukey
import numpy as np
import matplotlib.pyplot as plt
import mrdhelper
from pathlib import Path
import copy
import pyfftw
from editer import autopick_sensing_coils


# Read config
with open('config.toml', 'r') as cf:
    cfg = rtoml.load(cf)

DATA_ROOT = cfg['DATA_ROOT']
DATA_DIR = cfg['data_folder']
raw_file = cfg['raw_file']
remove_os = cfg['saving']['remove_os']
f_pt = cfg['pilottone']['pt_freq']
ismrmrd_data_fullpath, ismrmrd_noise_fullpath = mrdhelper.siemens_mrd_finder(DATA_ROOT, DATA_DIR, raw_file)

# Read the data in
print(f'Reading {ismrmrd_data_fullpath}...')
with ismrmrd.Dataset(ismrmrd_data_fullpath) as dset:

    n_acq = dset.number_of_acquisitions()
    print(f'There are {n_acq} acquisitions in the file. Reading...')

    acq_list = []
    for ii in range(n_acq):
        acq_list.append(dset.read_acquisition(ii))

    n_wf = dset.number_of_waveforms()
    print(f'There are {n_wf} waveforms in the dataset. Reading...')

    wf_list = []
    for ii in range(n_wf):
        wf_list.append(dset.read_waveform(ii))
    
    hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())

# get the k-space trajectory based on the metadata hash.
traj_name = hdr.userParameters.userParameterString[1].value

# load the .mat file containing the trajectory
traj = loadmat(os.path.join(DATA_ROOT, DATA_DIR, traj_name), squeeze_me=True)

dt = float(traj['param']['dt'])
pre_discard = int(traj['param']['pre_discard'])

data = [arm.data[:,:] for arm in acq_list]

data = np.array(data)
data = np.transpose(data, axes=(2, 0, 1))


# %%
mri_coils = np.arange(15, dtype=int)
sensing_coils = np.array([15, 16, 17], dtype=int)
coil_name = []

for clbl in hdr.acquisitionSystemInformation.coilLabel:
    coil_name.append(clbl.coilName)

coil_name = np.asarray(coil_name)

print(f"Coils to be used as sniffers: {coil_name[sensing_coils.astype(int)]}")

f0 = hdr.experimentalConditions.H1resonanceFrequency_Hz

ksp_measured = data[:,:,mri_coils]

# %% [markdown]
# # Save the processed k-space


ksp_win = tukey(2*data.shape[0], alpha=0.1)
ksp_win = ksp_win[(data.shape[0]):,None,None]
ksp_ptsubbed = data[:,:,mri_coils]*ksp_win # kspace filtering to remove spike at the end of the acquisition

# Read the noise data in
print(f'Reading {ismrmrd_noise_fullpath}...')
with ismrmrd.Dataset(ismrmrd_noise_fullpath) as dset_noise:
    n_cal_acq = dset_noise.number_of_acquisitions()
    print(f'There are {n_cal_acq} acquisitions in the file. Reading...')

    cal_list = []
    for ii in range(n_cal_acq):
        cal_list.append(dset_noise.read_acquisition(ii))

noise_list = []

for cal_ in cal_list:
    if cal_.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
        noise_list.append(cal_.data)

noise = np.transpose(np.asarray(noise_list), (1,0,2)).reshape((noise_list[0].shape[0], -1))[mri_coils,:]

if cfg['pilottone']['prewhiten']:
    from reconstruction.coils import apply_prewhitening, calculate_prewhitening

    print('Prewhitening the raw data...')
    dmtx = calculate_prewhitening(noise)

    ksp_ptsubbed = apply_prewhitening(np.transpose(ksp_ptsubbed, (2,0,1)), dmtx).transpose((1,2,0))

    dmtx2 = calculate_prewhitening(apply_prewhitening(noise, dmtx))

    _,axs = plt.subplots(1,2)
    axs[0].imshow(np.abs(dmtx))
    axs[1].imshow(np.abs(dmtx2))
    plt.show()

if cfg['pilottone']['discard_badcoils']:
    mri_coils2, _ = autopick_sensing_coils(ksp_measured, f_emi=f0-f_pt, bw_emi=100e3, bw_sig=200e3, f_samp=1/dt, n_sensing=5)
    ksp_ptsubbed = ksp_ptsubbed[:,:,mri_coils2]
    mri_coils = mri_coils[mri_coils2]
    
n_samp = ksp_ptsubbed.shape[0]

if remove_os:
    ksp_ptsubbed = pyfftw.byte_align(ksp_ptsubbed)

    keepOS = np.concatenate([np.arange(n_samp // 4), np.arange(n_samp * 3 // 4, n_samp)])
    ifft_ = pyfftw.builders.ifft(ksp_ptsubbed, n=n_samp, axis=0, threads=32, planner_effort='FFTW_ESTIMATE')
    ksp_ptsubbed = ifft_()

    fft_ = pyfftw.builders.fft(ksp_ptsubbed[keepOS, :, :], n=keepOS.shape[0], axis=0, threads=32, planner_effort='FFTW_ESTIMATE')
    ksp_ptsubbed = fft_()
    n_samp = n_samp // 2


output_dir_fullpath = os.path.join(DATA_ROOT, DATA_DIR, 'raw', 'h5_proc')
output_data_fullpath = os.path.join(output_dir_fullpath, f"{ismrmrd_data_fullpath.split('/')[-1][:-3]}_reference.h5")
print('Saving to ' + output_data_fullpath)

Path.mkdir(Path(output_dir_fullpath), exist_ok=True)

# Add EDITER parameters to XML header.
new_hdr = copy.deepcopy(hdr)
new_hdr.acquisitionSystemInformation.coilLabel = [hdr.acquisitionSystemInformation.coilLabel[ch_i] for ch_i in mri_coils]
new_hdr.acquisitionSystemInformation.receiverChannels = len(new_hdr.acquisitionSystemInformation.coilLabel)

# Copy and fix acquisition objects
new_acq_list = []

for acq_i, acq_ in enumerate(acq_list):
    new_head = copy.deepcopy(acq_.getHead())
    new_head.active_channels = len(new_hdr.acquisitionSystemInformation.coilLabel)
    new_head.available_channels = len(new_hdr.acquisitionSystemInformation.coilLabel)
    if remove_os:
        new_head.number_of_samples = ksp_ptsubbed.shape[0]
        new_head.center_sample = pre_discard//2

    new_acq_list.append(ismrmrd.Acquisition(head=new_head, data=np.ascontiguousarray(ksp_ptsubbed[:,acq_i,:].squeeze().T.astype(np.complex64))))

with ismrmrd.Dataset(output_data_fullpath, create_if_needed=True) as new_dset:
    for acq_ in new_acq_list:
        new_dset.append_acquisition(acq_)

    for wave_ in wf_list:
        new_dset.append_waveform(wave_)

    new_dset.write_xml_header(ismrmrd.xsd.ToXML(new_hdr))



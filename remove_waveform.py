"""
Remove waveform by its ID from an ISMRMRD dataset. Especially useful when we want to replace custom waveforms.
Author: Bilal Tasdelen
"""
import rtoml
import os
import fnmatch
import h5py
import numpy as np
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Remove waveform by its ID from an ISMRMRD dataset. Especially useful when we want to replace custom waveforms.')
parser.add_argument('waveform_id', nargs='?', help='Waveform ID. Default is 1025.', default=1025, type=int)
parser.add_argument("-r", '--repack', help='Repack h5 file. May reduce file size, but takes some time.',action='store_true')
args = parser.parse_args()
repack_file = args.repack
waveform_id = args.waveform_id

with open('config.toml') as cf:
    cfg = rtoml.load(cf)
    DATA_ROOT = cfg['DATA_ROOT']
    DATA_DIR = cfg['data_folder']
    raw_file = cfg['raw_file']

data_dir_path = os.path.join(DATA_ROOT, DATA_DIR, 'raw/h5')
if raw_file.isnumeric():
    raw_file_ = fnmatch.filter(os.listdir(data_dir_path), f'meas_MID*{raw_file}*.h5')[0]
    ismrmrd_data_fullpath = os.path.join(data_dir_path, raw_file_)
elif raw_file.startswith('meas_MID'):
    raw_file_ = raw_file
    ismrmrd_data_fullpath = os.path.join(data_dir_path, raw_file)
else:
    print('Could not find the file. Exiting...')
    exit(-1)

with h5py.File(ismrmrd_data_fullpath, 'a') as f:
    wfs = f['/dataset/waveforms']
    wfs2 = []
    print(f'Finding and removing waveforms with ID {waveform_id}...')
    for row_i, wf in enumerate(wfs):
        if wf[0][8] != waveform_id:
            wfs2.append(wf)
            row_i-=1

    del f['/dataset/waveforms']

    f.create_dataset('/dataset/waveforms', maxshape=(None,), chunks=True, data=np.array(wfs2))

if repack_file is True:
    print('Repacking the file, this may take a while....')

    rpk_fname = '2rpk_' + raw_file_
    os.chdir(data_dir_path)
    os.rename(raw_file_, rpk_fname)
    print(subprocess.run(['h5repack', f'{rpk_fname}', f'{raw_file_}']))
    os.remove(rpk_fname)
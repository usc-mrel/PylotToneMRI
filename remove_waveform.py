"""
Remove waveform by its ID from an ISMRMRD dataset. Especially useful when we want to replace custom waveforms.
Author: Bilal Tasdelen
"""
import os
import h5py
import numpy as np
import subprocess
import argparse
from ui.selectionui import get_filepath
from PySide6.QtCore import QDir

parser = argparse.ArgumentParser(description='Remove waveform by its ID from an ISMRMRD dataset. Especially useful when we want to replace custom waveforms.')
parser.add_argument('waveform_id', nargs='?', help='Waveform ID. Default is 1025.', default=1025, type=int)
parser.add_argument("-r", '--repack', help='Repack h5 file. May reduce file size, but takes some time.',action='store_true')
args = parser.parse_args()
repack_file = args.repack
waveform_id = args.waveform_id

ismrmrd_data_fullpath = get_filepath(dir=QDir.homePath())
print(f'File selected: {ismrmrd_data_fullpath}')
raw_file_ = os.path.basename(ismrmrd_data_fullpath)
print(f'Raw file: {raw_file_}')
data_dir_path = os.path.dirname(ismrmrd_data_fullpath)
print(f'Data directory path: {data_dir_path}')

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
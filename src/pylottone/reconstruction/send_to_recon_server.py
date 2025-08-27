# %%
# # Setup the reconstruction

import argparse
import datetime
import os
from pathlib import Path
from types import SimpleNamespace

import rtoml

from pylottone.reconstruction import client
from pylottone.selectionui import get_multiple_filepaths


def main(ismrmrd_data_fullpath, cfg):
    DATA_DIR    = cfg['data_folder']
    recon_method    = cfg['reconstruction']['recon_type']
    server_port     = cfg['reconstruction']['server_port']
    show_images     = cfg['reconstruction']['show_images']
    output_folder   = cfg['reconstruction']['output_folder']
    raw_file_ = ismrmrd_data_fullpath.split('/')[-1]

    # Mapping from recon method to config file name for the server
    recon_config = {'viewsharing': 'simplenufft1arm',
                    'ttv': 'rtspiral_bart_tvrecon',
                    'xdgrasp': 'spiral_xdgrasp_recon'}

    # %% [markdown]
    # # Run reconstruction
    # Algorithms/reconstructions are run as modules by the server component, which implements the [MRD streaming protocol](https://ismrmrd.readthedocs.io/en/latest/mrd_streaming_protocol.html).  The server must be started separately and a client sends data to the server for processing.  The server can be started via command line by running `python main.py`.
    # 
    # If using Visual Studio Code, [launch.json](../.vscode/launch.json) includes a debug configuration to run the server and allow for interactive debugging.  Click on the "Run and Debug" tab on the top left, then click the green play button to start the desired config ("Start server"):
    # 
    # ![image-2.png](attachment:image-2.png)
    # 
    # After the server is started, begin the reconstruction by running [client.py](../client.py) on the input dataset.  The recon code/module is selected by using the `config` argument.  The output is stored in a separate MRD file under a dataset named with the current date/time.

    # %%
    # Run reconstruction

    print(f'Running reconstruction for {ismrmrd_data_fullpath} using {recon_method} reconstruction method.')
    Path(output_folder, DATA_DIR).mkdir(exist_ok=True, parents=True)
    outfilename = os.path.join(output_folder, DATA_DIR, f'{recon_method}_{raw_file_[:-3]}.mrd')

    # Specify client arguments for recon
    args = SimpleNamespace(**client.defaults)
    args.out_group = f"{recon_method}_{str(datetime.datetime.now())}"
    args.config   = recon_config[recon_method]  # Recon module to be used
    args.outfile  = outfilename
    args.filename = ismrmrd_data_fullpath
    # args.address = '10.136.17.174'
    args.port = server_port
    args.send_waveforms = True

    client.main(args)

    # %%
    if show_images:
        import h5py
        from pyArrView import av
        import ismrmrd
        import numpy as np

        # Assuming it is the latest subgroup in the dataset, load it.
        with h5py.File(outfilename, 'r') as d:
            dset_names = list(d.keys())
        group = dset_names[-1]

        with ismrmrd.Dataset(outfilename, group, False) as dset:
            subgroups = dset.list()
            # Images are organized by series number in subgroups that start with 'images_'
            imgGroups = [group for group in list(subgroups) if (group.find('image_') != -1)]
            print(f'Group {group} contains {len(imgGroups)} image series:')
            print(' ', '\n  '.join(imgGroups))

            # Assumes there is only one image series
            imgs = []
            n = dset.number_of_images(imgGroups[0])
            for ii in range(n):
                frame = dset.read_image(imgGroups[0], ii)
                imgs.append(np.squeeze(frame.data))

        imgs = np.flip(np.asarray(imgs), axis=2).transpose((2, 1, 0))

        av(imgs)
        input('Press any keys to end..')

def main_cli():
    # Check if filepaths are provided as arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--filepaths', nargs='+', help='List of filepaths to process')
    argparser.add_argument('-c', '--config', nargs='?', default='config.toml', help='Config file to be used during processing.')

    args = argparser.parse_args()

    with open(args.config, 'r') as cf:
        cfg = rtoml.load(cf)
    
    if args.filepaths:
        filepaths = args.filepaths
        print(f'Processing {len(filepaths)} files.')
        print(filepaths)
    else:
        # Get filepaths if not provided
        filepaths = get_multiple_filepaths(dir=os.path.join(cfg['DATA_ROOT'], cfg['data_folder'], 'raw'))

    for ismrmrd_data_fullpath in filepaths:
        main(ismrmrd_data_fullpath, cfg)

if __name__ == '__main__':
    main_cli()

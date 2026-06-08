# %%
# # Setup the reconstruction

import argparse
import datetime
import os
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import tomllib

try:
    from pylottone.selectionui import get_multiple_filepaths
except ImportError:
    get_multiple_filepaths = None


def _load_client_module():
    try:
        return import_module("pylottone.reconstruction.client")
    except ImportError as exc:
        raise ImportError(
            "MRD reconstruction support is not installed. Install with: pip install pylottone[mrd]"
        ) from exc




def main(ismrmrd_data_fullpath, cfg):
    client = _load_client_module()

    DATA_DIR    = cfg['data_folder']
    recon_method    = cfg['reconstruction']['recon_type']
    server_port     = cfg['reconstruction']['server_port']
    show_images     = cfg['reconstruction']['show_images']
    output_folder   = cfg['reconstruction']['output_folder']
    raw_file_ = ismrmrd_data_fullpath.split('/')[-1]

    recon_config = {'viewsharing': 'simplenufft1arm',
                    'ttv': 'rtspiral_bart_tvrecon',
                    'xdgrasp': 'spiral_xdgrasp_recon'}

    print(f'Running reconstruction for {ismrmrd_data_fullpath} using {recon_method} reconstruction method.')
    Path(output_folder, DATA_DIR).mkdir(exist_ok=True, parents=True)
    outfilename = os.path.join(output_folder, DATA_DIR, f'{recon_method}_{raw_file_[:-3]}.mrd')

    args = SimpleNamespace(**client.defaults)
    args.out_group = f"{recon_method}_{str(datetime.datetime.now())}"
    args.config   = recon_config[recon_method]
    args.outfile  = outfilename
    args.filename = ismrmrd_data_fullpath
    args.port = server_port
    args.send_waveforms = True

    client.main(args)

    if show_images:
        import h5py
        import ismrmrd
        import numpy as np
        from pyArrView import av

        with h5py.File(outfilename, 'r') as d:
            dset_names = list(d.keys())
        group = dset_names[-1]

        with ismrmrd.Dataset(outfilename, group, False) as dset:
            subgroups = dset.list()
            imgGroups = [group for group in list(subgroups) if (group.find('image_') != -1)]
            print(f'Group {group} contains {len(imgGroups)} image series:')
            print(' ', '\n  '.join(imgGroups))

            imgs = []
            n = dset.number_of_images(imgGroups[0])
            for ii in range(n):
                frame = dset.read_image(imgGroups[0], ii)
                imgs.append(np.squeeze(frame.data))

        imgs = np.flip(np.asarray(imgs), axis=2).transpose((2, 1, 0))

        av(imgs)
        input('Press any keys to end..')


def main_cli():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--filepaths', nargs='+', help='List of filepaths to process')
    argparser.add_argument('-c', '--config', nargs='?', default='config.toml', help='Config file to be used during processing.')

    args = argparser.parse_args()

    with open(args.config, 'rb') as cf:
        cfg = tomllib.load(cf)

    if args.filepaths:
        filepaths = args.filepaths
        print(f'Processing {len(filepaths)} files.')
        print(filepaths)
    else:
        if get_multiple_filepaths is None:
            raise ImportError(
                "UI dependencies not installed. Either provide filepaths with -f or install with: pip install pylottone[ui]"
            )
        filepaths = get_multiple_filepaths(dir=os.path.join(cfg['DATA_ROOT'], cfg['data_folder'], 'raw'))

    for ismrmrd_data_fullpath in filepaths:
        main(ismrmrd_data_fullpath, cfg)


if __name__ == '__main__':
    main_cli()

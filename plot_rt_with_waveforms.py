from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import scipy.io as sio
import h5py
import ismrmrd
from ui.selectionui import get_filepath, get_selection

def prep_1mov4navs_figure2(img, t_l1, l1, t_l2, l2, t_l3, l3, t_l4, l4):
    '''Creates the axes with the right sizes, and prepares the background for blipping. Returns the figure and the artists to be updated.'''
    # Colors for the Boxes and Texts
    
    fn = 'DejaVu Sans'
    fs = 16
    
    Nplots = 4
    
    # Padding and the Grid Size
    y_padding = 30
    x_padding_left = 150
    x_padding_right = 10
    im_scale = 3
    Ny2, Nx2, _ = img.shape
    Ny2 *= im_scale
    Nx2 *= im_scale
    
    # Width and Height Accordingly
    h_pl = (Nx2 - (Nplots - 1) * y_padding) / Nplots
    w_pl = 600
    W = Nx2 + x_padding_left + x_padding_right + w_pl
    H = Ny2 + (Nplots) * y_padding
    
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    # Set the outer figure
    fig.set_size_inches(W/100, H/100)
    plt.clf()
    
    # My image
    ax0 = fig.add_axes([0, 2*y_padding/H, Nx2/W, Ny2/H])
    imgax = ax0.imshow(img[:,:,0], cmap='gray', vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    ax0.axis('off')
    line_lp, = ax0.plot([], [], linestyle=':', linewidth=2.5, color='r')

    # Line Profiles
    
    # Line profile for SMS1, Slice1
    ax1 = fig.add_axes([(Nx2 + x_padding_left) / W, (5*y_padding + 3*h_pl) / H, w_pl / W, h_pl / H])
    ax1.plot(t_l1, l1, linewidth=2)

    ax1.legend(['Cardiac Pilot Tone'], loc='lower right')
    ax1.set_ylabel('Cardiac\ndisplacement', fontname=fn)
    ax1.axes.yaxis.set_ticklabels([])

    # Line profile for SMS1, Slice2
    ax2 = fig.add_axes([(Nx2 + x_padding_left) / W, (4*y_padding + 2*h_pl) / H, w_pl / W, h_pl / H])
    ax2.plot(t_l2, l2, 'r', linewidth=2)

    ax2.legend(['ECG'], loc='lower right')
    ax2.set_ylabel('ECG\namplitude')
    ax2.axes.yaxis.set_ticklabels([])
    # Line profile for Respiratory Pilot Tone
    ax3 = fig.add_axes([(Nx2 + x_padding_left) / W, (3*y_padding + 1*h_pl) / H, w_pl / W, h_pl / H])
    ax3.plot(t_l3, l3, linewidth=2)
    ax3.legend(['Respiratory Pilot Tone'], loc='lower right')
    ax3.set_ylabel('Respiratory\ndisplacement')
    ax3.axes.yaxis.set_ticklabels([])

    # Line profile for Liver Dome Movement
    ax4 = fig.add_axes([(Nx2 + x_padding_left) / W, (2*y_padding) / H, w_pl / W, h_pl / H])
    ax4.plot(t_l4, l4, 'r', linewidth=2)
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Pixel shift')
    ax4.legend(['Liver Dome Movement'], loc='lower right')

    # Add vertical lines at t_step
    line0 = ax1.axvline(0, color='k', linestyle='--', linewidth=2)
    line1 = ax2.axvline(0, color='k', linestyle='--', linewidth=2)
    line2 = ax3.axvline(0, color='k', linestyle='--', linewidth=2)
    line3 = ax4.axvline(0, color='k', linestyle='--', linewidth=2)
    
    # Adjust font sizes
    for ax in [ax1, ax2, ax3, ax4]:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fs)
    
    return fig, (imgax, line0, line1, line2, line3, line_lp)

def update_mov(frame, img, t_sel, lines, lp):
    '''Updates the artists for the given'''
    lines[0].set_data(img[:, :, frame])
    lines[-1].set_data(lp[:,0], lp[:,1])
    lines[1].set_xdata([t_sel[frame]])
    lines[2].set_xdata([t_sel[frame]])
    lines[3].set_xdata([t_sel[frame]])
    lines[4].set_xdata([t_sel[frame]])
    return lines


if __name__ == '__main__':
    
    # Load data from .mat file
    # filename = 'vol0820_spiral_0.40V_spiral_basic_240328_1639.mat'
    filepath = get_filepath()
    if not filepath:
        print('No file selected, exiting...')
        exit()
        
    # Set video parameters
    video_name = f'{filepath[:-4]}.mp4'
    framerate = 20
    ecg_sign = -1
    t_sel = [1, 10]  # [s]

    if filepath.endswith('.h5') or filepath.endswith('.mrd'):
        # If run multiple times, the recon.mrd file will have multiple reconstructed images
        # Find the most recent recon run
        with h5py.File(filepath, 'r') as d:
            dset_names = list(d.keys())
            print(f'File {filepath} contains {len(dset_names)} groups (reconstruction runs):')
            # print(' ', '\n  '.join(dset_names))

        group = get_selection(dset_names)
        video_name = f'{filepath[:-4]}_{group}.mp4'
        wf_list = []
        with ismrmrd.Dataset(filepath, group, False) as dset:
            subgroups = dset.list()

            # Images are organized by series number in subgroups that start with 'images_'
            imgGroups = [group for group in list(subgroups) if (group.find('image_') != -1)]
            print(f'Group {group} contains {len(imgGroups)} image series:')
            print(' ', '\n  '.join(imgGroups))

            # Read and append images
            imgs = []
            time_stamps = []
            n = dset.number_of_images(imgGroups[0])
            for ii in range(n):
                frame = dset.read_image(imgGroups[0], ii)
                imgs.append(np.squeeze(frame.data))
                time_stamps.append(frame.acquisition_time_stamp)
            img_0 = dset.read_image(imgGroups[0], 0)

            n_wf = dset.number_of_waveforms()

            for ii in range(n_wf):
                wf_list.append(dset.read_waveform(ii))

        # =======================================
        ## Extract Pilot Tone navigator waveforms
        # =======================================

        resp_waveform = []
        pt_card_triggers = []
        timestamp0 = 0

        for wf in wf_list:
            if wf.getHead().waveform_id == 1025:
                resp_waveform = wf.data[0,:]
                cardiac_waveform = wf.data[1,:]
                # pt_card_triggers = ((wf.data[2,:]-2**31) != 0).astype(int)#np.round(((wf.data[2,:] - 2**31)/2**31)).astype(int)
                timestamp0 = wf.time_stamp
                pt_sampling_time = wf.getHead().sample_time_us*1e-6
                break
        time_pt = np.arange(resp_waveform.shape[0])*pt_sampling_time

        # =======================================
        ## Process ECG waveform
        # =======================================

        ecg_waveform = []
        ecg_trigs = []
        wf_init_timestamp = 0
        for wf in wf_list:
            if wf.getHead().waveform_id == 0:
                ecg_waveform.append(wf.data[1,:])
                ecg_trigs.append(wf.data[4,:])
                if wf_init_timestamp == 0:
                    wf_init_timestamp = wf.time_stamp
                    ecg_sampling_time = wf_list[0].getHead().sample_time_us*1e-6 # [us] -> [s]

        ecg_waveform = ecg_sign*(np.asarray(np.concatenate(ecg_waveform, axis=0), dtype=float)-2048)
        ecg_waveform = ecg_waveform/np.percentile(ecg_waveform, 99.9)
        ecg_trigs = (np.concatenate(ecg_trigs, axis=0)/2**14).astype(int)
        time_ecg = np.arange(ecg_waveform.shape[0])*ecg_sampling_time - (timestamp0 - wf_init_timestamp)*1e-3

        time_frame = np.array(time_stamps, dtype=float)*2.5e-3
        time_frame -= time_frame[0]
        # Update frame rate
        framerate = 1/(np.mean(np.diff(time_frame)))
        imgs = np.flip(np.asarray(imgs), axis=2).transpose((2, 1, 0))

        import respiratory_from_image

        time_imresp, resp_imnav, lp = respiratory_from_image.main(filepath, group)
        # lp = np.array(((imgs.shape[0]-lp[0,0]),()))

        # Time selection

        I_sel_I = ((time_frame > t_sel[0]) & (time_frame < t_sel[1])).squeeze()
        I_sel_l1 = ((time_ecg > t_sel[0]) & (time_ecg < t_sel[1])).squeeze()
        I_sel_l2 = ((time_pt > t_sel[0]) & (time_pt < t_sel[1])).squeeze()
        I_sel_l3 = ((time_pt > t_sel[0]) & (time_pt < t_sel[1])).squeeze()
        I_sel_l4 = ((time_imresp > t_sel[0]) & (time_imresp < t_sel[1])).squeeze()

        t_sel_I = time_frame[I_sel_I] - t_sel[0]
        t_sel_l1 = time_ecg[I_sel_l1] - t_sel[0]
        t_sel_l2 = time_pt[I_sel_l2] - t_sel[0]
        t_sel_l3 = time_pt[I_sel_l3] - t_sel[0]
        t_sel_l4 = time_imresp[I_sel_l4] - t_sel[0]

        pass


    elif filepath.endswith('.mat'):
        mat_data = sio.loadmat(os.path.join('output_recons', filepath))
        imgs = np.abs(mat_data['img_emicorr'].squeeze())

        cardiac_waveform = mat_data['recon_params']['cardiac_nav'][0][0]
        ecg_waveform = mat_data['ecg_wave'][0]
        resp_waveform = mat_data['recon_params']['respiratory_nav'][0][0][:,0]
        # # Optional cropping
        # if False:
        #     I = centered_crop(I, (56, 56, 0, 0))  # You would need to define centered_crop

        # Extract line profile
        # ll, lp = draw_profile_(I)  # Assume draw_profile_ is defined
        # l = ll[0]
        # resp_imnav = extract_nav_from_profile(l)  # Assume extract_nav_from_profile is defined

        # Timeframes
        time_frame = mat_data['recon_params']['t_nav'][0][0].T[7:-7:17]

        # Time selection

        I_sel_I = ((time_frame > t_sel[0]) & (time_frame < t_sel[1])).squeeze()
        I_sel_l1 = ((mat_data['t_ecg_'] > t_sel[0]) & (mat_data['t_ecg_'] < t_sel[1])).squeeze()
        I_sel_l2 = ((mat_data['recon_params']['t_nav'][0][0] > t_sel[0]) & (mat_data['recon_params']['t_nav'][0][0] < t_sel[1])).squeeze()
        I_sel_l3 = ((mat_data['recon_params']['t_nav'][0][0] > t_sel[0]) & (mat_data['recon_params']['t_nav'][0][0] < t_sel[1])).squeeze()
        I_sel_l4 = ((mat_data['recon_params']['t_nav'][0][0] > t_sel[0]) & (mat_data['recon_params']['t_nav'][0][0] < t_sel[1])).squeeze()

        t_sel_I = time_frame[I_sel_I] - t_sel[0]
        t_sel_l1 = mat_data['t_ecg_'][0,I_sel_l1] - t_sel[0]
        t_sel_l2 = mat_data['recon_params']['t_nav'][0][0][0,I_sel_l2] - t_sel[0]
        t_sel_l3 = mat_data['recon_params']['t_nav'][0][0][0,I_sel_l3] - t_sel[0]
        t_sel_l4 = mat_data['recon_params']['t_nav'][0][0][0,I_sel_l4] - t_sel[0]
    else:
        print('Unsupported file extension.')

    Nframes = np.sum(I_sel_I)
    Iv = []


    fig, lines = prep_1mov4navs_figure2(imgs[:, :, I_sel_I],
                                t_sel_l2, cardiac_waveform[I_sel_l2],
                                t_sel_l1, ecg_waveform[I_sel_l1],
                                t_sel_l3, -resp_waveform[I_sel_l3] / np.percentile(np.abs(resp_waveform[I_sel_l3]), 99.9),
                                t_sel_l4, resp_imnav[I_sel_l4])
                                # t_sel_l3, -resp_waveform[I_sel_l3] / np.percentile(np.abs(resp_waveform[I_sel_l3]), 99.9))

    ani = animation.FuncAnimation(fig, partial(update_mov, img=imgs[:, :, I_sel_I], t_sel=t_sel_I, lines=lines, lp=lp), 
                                  frames=range(Nframes), interval=1e3/framerate, blit=True)
    MWriter = animation.FFMpegWriter(fps=framerate)
    plt.show()
    print(f'Saving in {video_name}...')
    ani.save(video_name, writer=MWriter)


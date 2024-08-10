import h5py
import ismrmrd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from pilottone import qint
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from ui.selectionui import get_selection, get_filepath
from ui import filedialog
import hyperspy.api as hs

def extract_nav_from_profile(line_profile, ):
    n_time = line_profile.shape[1]


    # Find peak method
    line_profile -= np.percentile(line_profile, 0.1, axis=0)
    line_profile /= np.percentile(line_profile, 99.9, axis=0)
    peak_locs = []
    for c_i in range(line_profile.shape[1]):
        p, _ = find_peaks(line_profile[:,c_i], prominence=0.2, distance=line_profile.shape[0])

        peak_locs.append(p)

    # Fill missing points with preceding
    rough_pks = np.zeros((n_time,))
    for c_i in range(n_time):
        if len(peak_locs[c_i]) == 0 and c_i > 1 and c_i < n_time:
            rough_pks[c_i] = rough_pks[c_i-1]
        else:
            rough_pks[c_i] = peak_locs[c_i][0]

    # Old method
    first_est = rough_pks.astype(int)

    nav = np.zeros(first_est.shape)
  
    for c_i in range(first_est.shape[0]):
        pos1 = first_est[c_i]
        p,_,_ = qint(line_profile[pos1-1, c_i], line_profile[pos1, c_i], line_profile[pos1+1, c_i])
        nav[c_i] = pos1 + p
    
    nav = median_filter(np.max(nav) - nav, size=(3,), axes=(0,))
    return nav

def get_line_profile(imgs: npt.NDArray[np.float32], 
                     resolution: npt.ArrayLike=[1.0, 1.0],
                     timestep: float = 1.0) -> npt.NDArray[np.float32]:
    
    ff, ax = plt.subplots()

    def profile_changed(obj):
        ax.clear()
        ax.imshow(obj.data.T, cmap='gray')
        ax.set_axis_off()
        ff.canvas.draw()

    
    # Fill the spatial and temporal scales
    s = hs.signals.Signal2D(imgs)
    s.axes_manager[0].name = 'Time'
    s.axes_manager[0].unit = 'ms'
    s.axes_manager[0].scale = timestep
    s.axes_manager[1].name = 'x'
    s.axes_manager[1].unit = 'mm'
    s.axes_manager[1].scale = resolution[0]
    s.axes_manager[2].name = 'y'
    s.axes_manager[2].unit = 'mm'
    s.axes_manager[2].scale = resolution[1]

    # s.calibrate(x0=1, y0=1, x1=2, y1=2, new_length=1.87, units="mm", interactive=False)
    line_roi = hs.roi.Line2DROI(x1=100, y1=100, x2=100, y2=120, linewidth=8)
    s.plot(navigator='spectrum', colorbar=False, axes_ticks=False)
    # hs.plot.plot_images(s)
    profile = line_roi.interactive(s, color='green', navigation_signal='same')
    ax.imshow(profile.data.T, cmap='gray')
    ax.set_axis_off()
    profile.events.data_changed.connect(profile_changed)

    def on_button_press(event):
        plt.close('all')

    bt_ax = ff.add_axes([0.7, 0.05, 0.2, 0.075])
    bt = mpl.widgets.Button(bt_ax, 'Save Profile')
    bt.on_clicked(on_button_press)
    plt.show()

    return profile.data.T

def normalize_to_uint32(arr: npt.NDArray):
    arr -= np.min(arr) # 0 to max
    arr /= np.max(arr) # 0 to 1
    arr *= (2**32-1) # 0 to 2^32-1
    return arr.astype(np.uint32)

if __name__ == '__main__':
    # outfilename = f'output_recons/vol0902_20240611/viewsharing_meas_MID00175_FID15580_pulseq2D_fire_spiralga_400mV_24MHz_editer.mrd'
    # Display the reconstructed images
    outfilename = get_filepath()
    if not outfilename:
        print('No file selected, exiting...')
        exit()
    # If run multiple times, the recon.mrd file will have multiple reconstructed images
    # Find the most recent recon run
    with h5py.File(outfilename, 'r') as d:
        dset_names = list(d.keys())
        print(f'File {outfilename} contains {len(dset_names)} groups (reconstruction runs):')
        print(' ', '\n  '.join(dset_names))

    group = get_selection(dset_names)

    dset = ismrmrd.Dataset(outfilename, group, False)
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
    dset.close()

    time_frame = np.array(time_stamps, dtype=float)*2.5e-3
    time_frame -= time_frame[0]
    imgs = np.flip(np.asarray(imgs), axis=2).transpose((0, 2, 1))
    resolution = img_0.field_of_view[0:2]/np.array(img_0.matrix_size[1:])
    time_step = time_frame[1] - time_frame[0]

    mpl.use('Qt5Agg')
    line_profile = get_line_profile(imgs, resolution, time_step)
   
    resp_nav = extract_nav_from_profile(line_profile)

    def on_button_press(event, resp_nav, time_stamps, time_step, outfilename, group):
        # Concat, and normalize pt waveforms.
        import ctypes
        resp_nav = normalize_to_uint32(resp_nav) #((resp_nav/np.max(np.abs(resp_nav) - 0.5)*(2**31-1)) + 2**31).astype(np.uint32)

        nav_wf = ismrmrd.waveform.Waveform.from_array(resp_nav[None,:])
        nav_wf._head.sample_time_us = ctypes.c_float(time_step*1e6)
        nav_wf._head.waveform_id = ctypes.c_uint16(1029)
        nav_wf._head.time_stamp = int(time_stamps[0] - (time_stamps[1] - time_stamps[0])//2)

        with ismrmrd.Dataset(outfilename, group, False) as dset:
            dset.append_waveform(nav_wf)
        print('Done writing the waveform.')
        plt.close()
    
    ff = plt.figure()
    plt.plot(time_frame, resp_nav)
    plt.title('Extracted navigator.')

    bt_ax = ff.add_axes([0.7, 0.05, 0.2, 0.075])
    bt = mpl.widgets.Button(bt_ax, 'Save Navigator')
    bt.on_clicked(lambda x: on_button_press(x, resp_nav, time_stamps, time_step, outfilename, group))
    plt.show()


    # np.save('output_recons/vol0902_20240611/respnav_meas_MID00175_FID15580_pulseq2D_fire_spiralga_400mV_24MHz_editer', nav)
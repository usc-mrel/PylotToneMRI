import h5py
import ismrmrd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tkinter as tk
import numpy.typing as npt
from pilottone import qint
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

def selection_ui(dsetNames: list[str]) -> str:
    root = tk.Tk()
    var = tk.IntVar()
    root.title("Select a group from the dataset.")
    lb = tk.Listbox(root, selectmode=tk.SINGLE, height = len(dsetNames), width = 50) # create Listbox
    for x in dsetNames: lb.insert(tk.END, x)
    lb.pack() # put listbox on window
    lb.select_set(len(dsetNames)-1)
    btn = tk.Button(root,text="Select Group",command=lambda: var.set(1))
    btn.pack()
    # root.mainloop()
    btn.wait_variable(var)
    group = lb.get(lb.curselection()[0])
    root.destroy()
    return group

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

def get_line_profile(imgs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    ff, ax = plt.subplots()
    def profile_changed(obj):
        ax.clear()
        ax.imshow(obj.data.T, cmap='gray')
        ax.set_axis_off()
        ff.canvas.draw()

    import hyperspy.api as hs
    
    s = hs.signals.Signal2D(imgs)
    s.axes_manager[0].name = 'Time'
    s.axes_manager[0].unit = 'ms'
    s.axes_manager[0].scale = 5.5
    s.axes_manager[1].name = 'x'
    s.axes_manager[1].unit = 'mm'
    s.axes_manager[2].name = 'y'
    s.axes_manager[2].unit = 'mm'

    print(s)
    s.calibrate(x0=1, y0=1, x1=2, y1=2, new_length=1.87, units="mm", interactive=False)
    line_roi = hs.roi.Line2DROI(x1=100, y1=100, x2=100, y2=120, linewidth=4)
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


if __name__ == '__main__':
    outfilename = f'output_recons/vol0902_20240611/viewsharing_meas_MID00175_FID15580_pulseq2D_fire_spiralga_400mV_24MHz_editer.mrd'
    # Display the reconstructed images

    # If run multiple times, the recon.mrd file will have multiple reconstructed images
    # Find the most recent recon run
    with h5py.File(outfilename, 'r') as d:
        dsetNames = list(d.keys())
        print('File %s contains %d groups (reconstruction runs):' % (outfilename, len(dsetNames)))
        print(' ', '\n  '.join(dsetNames))

    group = selection_ui(dsetNames)


    dset = ismrmrd.Dataset(outfilename, group, False)
    subgroups = dset.list()

    # Images are organized by series number in subgroups that start with 'images_'
    imgGroups = [group for group in list(subgroups) if (group.find('image_') != -1)]
    print('Group %s contains %d image series:' % (group, len(imgGroups)))
    print(' ', '\n  '.join(imgGroups))

    mpl.use('QtAgg')
    # Show images
    # fig, axs = plt.subplots(1, len(imgGroups), squeeze=False)
    imgs = []
    n = dset.number_of_images(imgGroups[0])
    for ii in range(n):
        imgs.append(np.squeeze(dset.read_image(imgGroups[0], ii).data))
    
    dset.close()

    imgs = np.flip(np.asarray(imgs), axis=2).transpose((0, 2, 1))
    line_profile = get_line_profile(imgs)
   
    nav = extract_nav_from_profile(line_profile)
    plt.figure()
    plt.plot(nav)
    plt.title('Extracted navigator.')
    plt.show()
    np.save('output_recons/vol0902_20240611/respnav_meas_MID00175_FID15580_pulseq2D_fire_spiralga_400mV_24MHz_editer', nav)
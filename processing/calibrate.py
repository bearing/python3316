import numpy as np
import matplotlib.pyplot as plt


def main():
    correction = np.load('th_uncalib_Oct31_flood.npy')
    mm = np.load('step_run_1t2cm_Nov3.npy')  # 6A looks more right
    mm5 = np.load('step_run_9t10cm_Nov3.npy')
    # 'full_run_pos6A_3.5t7_Nov3.npy' is 20 minutes,not a full run

    sum_bg = np.sum(correction)
    mean_bg = np.mean(correction)
    print("Total number of counts in flood:", sum_bg)
    print("Mean number of counts in flood:", mean_bg)
    gain_correction = 1.0 * mean_bg / correction
    print("Gain correction shape:", gain_correction.shape)

    gain_correction[35, 11:24] = 0
    # gain_correction[:12, 11:24] = 0
    # gain_correction[36:, 25:37] = 0
    # gain_correction[:] = 1
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(correction.T, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal')
    ax2.imshow(np.floor(gain_correction * mm).T, cmap='jet', origin='upper', interpolation='nearest', aspect='equal')
    ax3.imshow(np.floor(gain_correction * mm5).T, cmap='jet', origin='upper', interpolation='nearest', aspect='equal')

    # fig.colorbar(im, ax=ax2)
    fig.tight_layout()
    plt.show()


def waterfall_plot_position():
    base_path = '/home/proton/repos/python3316/processing/images/step_run_'
    end_path = 'cm_Feb10.npz'

    fnum = np.arange(9)
    step = fnum[0] + 1
    fname = base_path + str(step) + "t" + str(step + 1) + end_path
    data = np.load(fname)['module_histograms']
    mods, bins = data.shape
    waterfall = np.zeros([(mods + 1) * fnum.size, bins])

    gid = 0
    waterfall[gid:(gid+mods), :] = data/np.max(data, axis=1)[:, np.newaxis]

    for global_ind in fnum[1:]:
        gid += 1
        step = global_ind + 1
        fname = base_path + str(step) + "t" + str(step + 1) + end_path
        data = np.load(fname)['module_histograms']

        ind = gid * (mods + 1)
        waterfall[ind:(ind + mods), :] = data / np.max(data, axis=1)[:, np.newaxis]

    plt.imshow(waterfall, cmap='jet', origin='upper', interpolation='nearest', aspect='equal')
    plt.axis('off')
    plt.colorbar()
    plt.title('Waterfall Plot of Single Cm steps')
    plt.show()

    # np.savez(filename, dyn_pmt_gains=self.dyn_pmt_gains,
    #         dyn_mod_gains=self.dyn_mod_gains,
    #         pmt_histograms=self.pmt_histograms,
    #         module_histograms=self.module_histograms,
    #         image_list=self.image_list)


def waterfall_plot():
    base_path = '/home/proton/repos/python3316/processing/images/step_run_'
    end_path = 'cm_Feb10.npz'

    fnum = np.arange(9)

    step = fnum[0] + 1
    fname = base_path + str(step) + "t" + str(step + 1) + end_path
    data = np.load(fname)['module_histograms']

    mods, bins = data.shape
    # waterfall = np.zeros([(mods + 1) * fnum.size, bins])

    n_meas = fnum.size
    mid = np.arange(mods)  # module id i.e. offset
    waterfall = np.zeros([mods * (n_meas + 1), bins])

    # waterfall[mid[0]::(n_meas+1)] = data/np.max(data, axis=1)[:, np.newaxis]  # Normalized
    waterfall[mid[0]::(n_meas + 1)] = data

    for meas in fnum[1:]:
        step = meas + 1
        fname = base_path + str(step) + "t" + str(step + 1) + end_path
        data = np.load(fname)['module_histograms']

        # waterfall[mid[meas]::(n_meas + 1)] = data / np.max(data, axis=1)[:, np.newaxis]  # Normalized
        waterfall[mid[meas]::(n_meas + 1)] = data

    ax = plt.axes()
    extent = [0, 8, 0, mods * (n_meas+1)]

    waterfall[waterfall > 0] = np.log(waterfall[waterfall > 0])
    # waterfall[waterfall > 0] = -1 * np.log(waterfall[waterfall > 0]) # Normalized
    plt.imshow(waterfall, cmap='jet', origin='upper', interpolation='nearest', aspect='auto', extent=extent)
    ax.set_yticks([])
    # plt.axis('off')
    plt.colorbar()
    plt.title('Unnormalized Waterfall Plot of Single Cm steps')
    plt.show()


def flood_map_separate(fname):
    pass


def sanity_check():
    arr = np.arange(48 * 48).reshape(48, 48)
    for sid in np.arange(16):
        row = sid//4
        col = sid%4
        arr[(row * 12): (row + 1) * 12, (col * 12): (col + 1) * 12] = sid
    plt.imshow(arr, cmap='viridis')
    plt.show()


if __name__ == "__main__":
    waterfall_plot()

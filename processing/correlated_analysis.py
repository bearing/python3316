import tables
import io
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from processing.calibration_values import load_calibration

# '/Users/justinellin/Desktop/Davis 2020/Tuesday/2020-10-06-1503.h5'
# -> beam axis is +x-hat, away from earth is +y-hat


class events_recon(object):

    def __init__(self, filepath, calib=None, place='Davis'):
        if calib is None:  # calib should be a dict
            calib = load_calibration(place)

        self._pmts = calib['pmts']
        self.swapped = calib['swapped']
        self._pmt_swapped = calib['swapped_pmts']

        self.h5file = load_signals(filepath)
        self.histogram_bins = np.arange(0, 180000, 1000)
        self.orig_hist_size = np.array([100, 100])
        # self.histogram_bins = np.arange(0, 30000, 1000)  # For background
        self.gamma_events = 0
        self.filtered_gamma_events = 0

        self.pmt_gains = calib['pmt_gains']
        self.pmt_shifts = calib['pmt_shifts']
        self.module_gains = calib['module_gains']
        self.module_shifts = calib['module_shifts']

        self.alphas = calib['alpha_undistort']
        self.com_bins = np.linspace(-1.0, 1.0, 101)
        self.crude_crystal_cutsX = calib['crystal_x_edges']
        self.crude_crystal_cutsY = calib['crystal_y_edges']

        self.energy_coeff_a = calib['energy_a']
        self.energy_coeff_b = calib['energy_b']

    def projection(self, ch_start_index, mod_ref_index):  # ch_start_index should be 0-15 not 0-60
        # Ch_start index is the channel as read by the digitizer. Mod ref index is as defined globally (mod id)

        tables = self.h5file.root.event_data
        struct = np.array(tables[tables.col('mod_id') == ch_start_index], dtype=tables.description._v_dtype)
        filter = np.unique(struct['bid'], return_index=True)[1]  # this deals with a bug that
        # should be fixed eventually

        data = struct[filter]
        evts = data.size

        if evts < 1:
            return None, None

        energy_array = np.zeros([4, evts])  # evts are columns

        self.gamma_events += evts

        if self.swapped[mod_ref_index]:
            pmts_ch_map = self._pmt_swapped
            print("Swapped. Mod ref idx:", mod_ref_index, ". ch_start:", ch_start_index//4)
        else:
            pmts_ch_map = self._pmts

        # When facing the front of the detectors. This maps spatial position of PMTs to your cabling
        ul = pmts_ch_map[0][0]
        ur = pmts_ch_map[0][1]
        ll = pmts_ch_map[1][0]
        lr = pmts_ch_map[1][1]
        pmt_ch_map = np.array([ul, ur, ll, lr])

        estrs = ['E2', 'E4', 'E1', 'E3']

        for pmt_id, ch_id in enumerate(pmt_ch_map):
            # pmt_id = 0, 1, 2, 3
            energy_array[pmt_id] = self.pmt_gains[mod_ref_index, pmt_id] * data[estrs[pmt_id]] -\
                                   self.pmt_shifts[mod_ref_index, pmt_id]

        raw_sum = np.sum(energy_array, axis=0)
        Ex = ((energy_array[1] - energy_array[0]) + (energy_array[3]-energy_array[2]))/(1.0 * raw_sum)
        Ey = ((energy_array[1] - energy_array[3]) + (energy_array[0]-energy_array[2]))/(1.0 * raw_sum)

        # if self.alphas[mod_ref_index]:  # TODO: Add when not under time crunch
        #    # print("raw_Ex shape:", raw_Ex.shape)
        #    r = raw_Ex**2 + raw_Ey**2
        #    # print("r size", r.shape)
        #    Ex = (raw_Ex / (1 - self.alphas[mod_ref_index] * r)).ravel()
        #    # print("Ex shape:", Ex.shape)
        #    Ey = (raw_Ey / (1 - self.alphas[mod_ref_index] * r)).ravel()
        # else:
        #    Ex = raw_Ex
        #    Ey = raw_Ey

        original_hist = np.histogram2d(Ex, Ey, bins=[self.com_bins, self.com_bins])[0]

        self.orig_hist_size = np.array([self.com_bins.size, self.com_bins.size])

        energy_hist = np.histogram(raw_sum, bins=self.histogram_bins)[0]
        E1 = np.histogram(energy_array[0], bins=self.histogram_bins)[0]   # ul
        E2 = np.histogram(energy_array[1], bins=self.histogram_bins)[0]   # ur
        E3 = np.histogram(energy_array[2], bins=self.histogram_bins)[0]   # ll
        E4 = np.histogram(energy_array[3], bins=self.histogram_bins)[0]   # lr

        return [energy_hist, E1, E2, E3, E4], original_hist

    def projection_binned(self, ch_start_index, mod_ref_index, energy_filter=None, time_filter=None):
        # Ch_start index is the channel as read by the digitizer. Mod ref index is as defined globally (mod id)
        # print("Ch_start_index (mod): ", ch_start_index)
        # print("Mod_ref_index: ", mod_ref_index)

        if energy_filter is None:
            energy_filter = []

        tables = self.h5file.root.event_data
        struct = np.array(tables[tables.col('mod_id') == ch_start_index], dtype=tables.description._v_dtype)
        filter = np.unique(struct['bid'], return_index=True)[1]  # this deals with a bug that
        # should be fixed eventually

        valid_data = struct[filter]

        if time_filter is not None:
            t_arr = np.sort(np.array(time_filter))  # This should be in samples
            t_filter = (valid_data['rel_ts'] > t_arr[0]) & (valid_data['rel_ts'] < t_arr[1])
            data = valid_data[t_filter]
        else:
            data = valid_data

        evts = data.size
        if evts < 1:
            return None, None, None

        energy_array = np.zeros([4, evts])  # evts are columns

        self.gamma_events += evts

        if self.swapped[mod_ref_index]:
            pmts_ch_map = self._pmt_swapped
        else:
            pmts_ch_map = self._pmts

        # When facing the front of the detectors. This maps spatial position of PMTs to your cabling
        ul = pmts_ch_map[0][0]
        ur = pmts_ch_map[0][1]
        ll = pmts_ch_map[1][0]
        lr = pmts_ch_map[1][1]
        pmt_ch_map = np.array([ul, ur, ll, lr])

        # estrs = ['E2', 'E4', 'E1', 'E3']
        estrs = ['E1', 'E2', 'E3', 'E4']

        # for pmt_id, ch_id in enumerate(pmt_ch_map):
        for pmt_id, ch_idx in enumerate(pmt_ch_map):
            # pmt_id = 0, 1, 2, 3
            energy_array[pmt_id] = self.pmt_gains[mod_ref_index, pmt_id] * data[estrs[ch_idx]] -\
                                   self.pmt_shifts[mod_ref_index, pmt_id]

        orig_raw_sum = np.sum(energy_array, axis=0)
        origEx = ((energy_array[1] - energy_array[0]) + (energy_array[3] - energy_array[2])) / (1.0 * orig_raw_sum)
        origEy = ((energy_array[1] - energy_array[3]) + (energy_array[0] - energy_array[2])) / (1.0 * orig_raw_sum)

        # print("Negative values:", np.count_nonzero(orig_raw_sum<0))
        valid_ind = (orig_raw_sum > 0)

        valid_Ex = origEx[valid_ind]
        valid_Ey = origEy[valid_ind]
        valid_E_sum = orig_raw_sum[valid_ind]

        sum_mev = self.energy_coeff_a[mod_ref_index] * np.log(1.0 * valid_E_sum) - self.energy_coeff_b[mod_ref_index]

        if len(energy_filter) is not 0:
            if len(energy_filter) == 2:
                energy_filter.sort()
                filt_ind = (sum_mev > energy_filter[0]) & (sum_mev < energy_filter[1])
            else:
                filt_ind = (sum_mev > energy_filter[0])

            sum_en = valid_E_sum[filt_ind]
            Ex = valid_Ex[filt_ind]
            Ey = valid_Ey[filt_ind]
        else:
            sum_en = valid_E_sum
            Ex = valid_Ex
            Ey = valid_Ey

        # if ch_start_index//4 == 2:
        #    Ex += 0.05
        #    Ey += 0.05

        binX = np.linspace(-1.0, 1.0, 101)
        binY = np.linspace(-1.0, 1.0, 101)

        original_hist = np.histogram2d(Ex, Ey, bins=[binX, binY])[0]
        self.orig_hist_size = np.array([binX.size, binY.size])

        Ex_scaled = (Ex + 1) / 0.02
        Ey_scaled = (Ey + 1) / 0.02

        Ex_scaled[Ex_scaled > 100] = 99
        Ey_scaled[Ey_scaled > 100] = 99

        energy_hist = np.histogram(sum_en, bins=self.histogram_bins)[0]
        self.filtered_gamma_events += np.sum(energy_hist)

        crude_binning = np.histogram2d(Ex_scaled, Ey_scaled,
                                       bins=[self.crude_crystal_cutsX[mod_ref_index].ravel(),
                                             self.crude_crystal_cutsY[mod_ref_index].ravel()])[0]

        return energy_hist, original_hist, crude_binning


def load_signals(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def main():

    base_path = '/home/proton/repos/python3316/processing/processedW-48+48_10-08-'
    file = base_path + '1530.h5'  # Position 6, far

    # file = '/home/proton/repos/python3316/Data/2020-10-31-1704.h5'  # Berkeley measurement

    tst = events_recon(file)  # Davis
    # tst = events_recon(file, place='Berkeley')  # Berkeley

    fig, (ax1, ax2) = plt.subplots(1, 2)

    gl_mod_id = np.arange(16)
    berkeley_mod_id = np.arange(16)
    davis_mod_id = np.arange(16)[::-1]

    # Davis Measurements
    # mod_id = 2  # Gain way off
    mod_id = 0  # Swapped channels, LR ambiguity
    mod_ref_id = np.argwhere(davis_mod_id == mod_id)  # 13 global id
    print("Mod Ref ID:", mod_ref_id)  # 13 for gain, 6 for swap

    # Berkeley Measurement
    # mod_id = 9
    # mod_ref_id = mod_id

    eng, proj = tst.projection(mod_id, mod_ref_id)

    x = np.linspace(0, 100000, eng[0].size)
    # x = np.linspace(0, 30000, eng[0].size)  # for background overnight

    for idx, energy_hist in enumerate(eng):
        if idx == 0:
            ax1.step(x, energy_hist, label='mod' + str(mod_id))
        else:
            ax1.step(x, energy_hist, label='ch' + str(idx-1))
    mapped = proj
    tst.h5file.close()
    print("Total Gamma Events:", tst.gamma_events)
    ax1.set_yscale('log')
    ax1.legend(loc='best')
    im = ax2.imshow(mapped, cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')

    # fig.colorbar(im, ax=ax2)
    plt.show()


def main_binned():

    base_path = '/home/proton/repos/python3316/processing/processedW-48+48_10-08-'
    file = base_path + '1530.h5'  # Position 6, far

    tst = events_recon(file)  # Davis
    # tst = events_recon(file, place='Berkeley')  # Berkeley

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    gl_mod_id = np.arange(16)
    berkeley_mod_id = np.arange(16)
    davis_mod_id = np.arange(16)[::-1]

    # mod_id = 2  # Gain Problems
    mod_id = 9  # Gain way off
    # mod_id = 9  # Swapped channels, LR ambiguity
    mod_ref_id = np.argwhere(davis_mod_id == mod_id)  # 13 global id
    print("Mod Ref ID:", mod_ref_id)  # 13 for gain, 6 for swap

    eng, proj, binned = tst.projection_binned(mod_id, mod_ref_id, energy_filter=[], time_filter=np.array([0, 0.5]))

    x = np.linspace(0, tst.histogram_bins[-1], eng.size)
    x_e = 2 * np.log(x) - 17
    # x = np.linspace(0, 30000, eng[0].size)  # for background overnight

    ax1.step(x_e, eng, label='mod' + str(mod_id))

    mapped = proj
    tst.h5file.close()
    print("Total Gamma Events:", tst.gamma_events)
    print("Total Post-Filtered Events:", tst.filtered_gamma_events)
    ax1.set_yscale('log')
    ax1.set_xlim([2, 1.01 * np.max(x_e)])
    ax1.legend(loc='best')
    im = ax2.imshow(mapped, cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')
    ax3.imshow(binned, cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')

    # fig.colorbar(im, ax=ax2)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # main()  # unbinned
    main_binned()
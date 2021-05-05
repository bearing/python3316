import tables
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from processing.calibration_values import load_calibration
# from scipy import stats  # This will be necessary to do a per-pixel gain calibration
from scipy.ndimage import uniform_filter1d
from scipy.stats import binned_statistic_2d
from file_lists import run_mm_steps
# or single pixel spectrum (scipy.binned_statistic2d)


# This class has no time filtering, see event_recon_classes.py for that
class events_recon(object):
    evt_read_limit = 2 * 10 ** 4  # Number of events to read at a time # TODO: Convert to memory size using col itemsze
    pixels = 12

    def __init__(self, filepath, calib=None, place='Davis', mod_adc_max_bin=18000, mod_adc_bin_size=250,
                 pmt_adc_max_bin=18000, pmt_adc_bin_size=250):
        if calib is None:  # calib should be a dict
            calib = load_calibration(place)

        self._pmts = calib['pmts']
        self.swapped = calib['swapped']
        self._pmt_swapped = calib['swapped_pmts']

        self.h5file = load_signals(filepath)

        # self.histogram_bins = np.arange(0, 180000, 1000)
        self.mod_histogram_bins = np.arange(0, mod_adc_max_bin + mod_adc_bin_size, mod_adc_bin_size)
        self.pmt_histogram_bins = np.arange(0, pmt_adc_max_bin + pmt_adc_bin_size, pmt_adc_bin_size)
        # self.image_size = np.array([100, 100])

        self.gamma_events = 0
        self.filtered_gamma_events = 0

        self.pmt_gains = calib['pmt_gains']
        self.pmt_shifts = calib['pmt_shifts']
        self.module_gains = calib['module_gains']
        self.module_shifts = calib['module_shifts']

        # self.com_bins = np.linspace(-1.0, 1.0, 101)
        # ============ Histogram Objects ============
        self.crude_crystal_cutsX = calib['crystal_x_edges']
        self.crude_crystal_cutsY = calib['crystal_y_edges']
        self.pmt_histograms = [Hist1D(self.pmt_histogram_bins) for _ in np.arange(4)]
        self.E1, self.E2, self.E3, self.E4 = self.pmt_histograms  # aliases
        # 4 PMTs histograms
        self.module_energy_histogram = Hist1D(self.mod_histogram_bins)  # 1 module histograms
        self.mod_image = HistImage(self.crude_crystal_cutsX, self.crude_crystal_cutsY)

    def convert_adc_to_bins(self, ch_start_index, mod_ref_index, dynamic_pmt_gains=np.ones(4), dynamic_mod_gains=1,
                        filter_limits=None):  # this is version 2
        """Mask must be an array of linearized indices or a list of 4 linearized indices (PMT weighting)"""
        pmts = np.arange(4)
        tables = [0 for _ in pmts]

        if filter_limits is None:
            filter_limits = 0

        if len(filter_limits) >= 2:
            filter_limits.sort()

        for integer in pmts:
            folder = '/det' + str(int(ch_start_index + integer))
            tables[integer] = self.h5file.get_node('/', folder).EventData

        total_evts = tables[0].nrows
        if total_evts < 1:
            return None, None
        self.gamma_events += total_evts

        read_block_idx = 0
        energy_array = np.zeros([4, np.min([total_evts, self.evt_read_limit])])  # evts are columns
        # print("Hello!")
        if self.swapped[mod_ref_index]:
            pmts_ch_map = self._pmt_swapped
        else:
            pmts_ch_map = self._pmts

        # When facing the front of the detectors. This maps spatial position of PMTs to your cabling
        # ul = pmts_ch_map[0][0], ur = pmts_ch_map[0][1]
        # ll = pmts_ch_map[1][0], lr = pmts_ch_map[1][1]
        (ul, ur), (ll, lr) = pmts_ch_map[:]
        pmt_ch_map = np.array([ul, ur, ll, lr])

        start = 0
        end = 0

        self.mod_image.select(mod_ref_index)
        self.module_energy_histogram.clear()
        self.E1.clear()
        self.E2.clear()
        self.E3.clear()
        self.E4.clear()

        while end < total_evts or not read_block_idx:
            for pmt_id, ch_id in enumerate(pmt_ch_map):
                # pmt_id = 0, 1, 2, 3
                start = read_block_idx * self.evt_read_limit
                end = np.min([start + self.evt_read_limit, total_evts])
                energy_array[pmt_id, :(end-start)] = (self.pmt_gains[mod_ref_index, pmt_id] *
                                                      (tables[ch_id].cols.gate2[start:end] -
                                                       3.0 * tables[ch_id].cols.gate1[start:end]) -
                                                      self.pmt_shifts[mod_ref_index, pmt_id]) * dynamic_pmt_gains[pmt_id]

            all_sum = np.sum(energy_array[:, :(end-start)], axis=0) * dynamic_mod_gains

            e_bound_filter = (all_sum > filter_limits[0])
            if len(filter_limits) == 2:
                e_bound_filter &= (all_sum < filter_limits[1])

            sum_e = all_sum[e_bound_filter]
            valid_pmts = energy_array[:, :(end - start)][:, e_bound_filter]  # TODO: Does this work? It's weird

            Ex = ((valid_pmts[1] - valid_pmts[0]) + (valid_pmts[3] - valid_pmts[2])) / (1.0 * sum_e)
            Ey = ((valid_pmts[1] - valid_pmts[3]) + (valid_pmts[0] - valid_pmts[2])) / (1.0 * sum_e)

            Ex_scaled = (Ex + 1) / 0.02
            Ey_scaled = (Ey + 1) / 0.02

            Ex_scaled[Ex_scaled > 100] = 99.  # TODO: Should these be masked this way. Are they physical?
            Ey_scaled[Ey_scaled > 100] = 99.

            Ex_scaled[Ex_scaled < 0] = 1.
            Ey_scaled[Ey_scaled < 0] = 1.

            self.mod_image.fill(Ex_scaled, Ey_scaled)
            self.module_energy_histogram.fill(sum_e)

            self.E1.fill(valid_pmts[0])  # ul
            self.E2.fill(valid_pmts[1])  # ur
            self.E3.fill(valid_pmts[2])  # ll
            self.E4.fill(valid_pmts[3])  # lr

            read_block_idx += 1

        self.filtered_gamma_events += np.sum(self.module_energy_histogram.hist)
        histograms = [self.module_energy_histogram.data[1], self.E1.data[1], self.E2.data[1], self.E3.data[1], self.E4.data[1]]
        # plt.imshow(self.mod_image.data[2], cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')
        # plt.show()
        _, _, image_hist, raw_hist = self.mod_image.data  # TODO: Added
        return histograms, (image_hist, raw_hist)


class system_processing(object):

    def __init__(self, filepaths, place="Davis", **kwargs):
        if type(filepaths) == str:
            files = [filepaths]
        else:
            files = filepaths

        self.runs = []
        for file in files:
            self.runs.append(events_recon(file, place=place, **kwargs))

        self.system_id = np.arange(16)  # when facing front of detectors, upper left to upper right, then down
        if place == "Davis":
            self.mod_id = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 0, 1, 2, 3])
            # self.mod_id = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])  # The order they were
        # plugged in by channel id
            print("Place: Davis")
        if place == "Berkeley":
            self.mod_id = np.arange(16)
            print("Place: Berkeley")
        self.mod_num_energy_bins = self.runs[0].mod_histogram_bins.size - 1
        self.pmt_num_energy_bins = self.runs[0].pmt_histogram_bins.size - 1
        self.pmt_histograms = np.zeros([64, self.pmt_num_energy_bins])
        self.module_histograms = np.zeros([16, self.mod_num_energy_bins])
        self.dyn_pmt_gains = np.ones(64)
        self.dyn_mod_gains = np.ones(16)

        self.data_generated = False
        self.image_list = [0 for _ in np.arange(16)]  # list of module images by SID
        self.raw_image_list = [0 for _ in np.arange(16)]  # list of raw weights by SID
        self.filter_limits = [0]

    def _subprocess_mod_sum_histograms(self, rid, sid, **kwargs):  # raw_id (digitizer), system_id (global labeling)
        total_energy_spectra = np.zeros(self.mod_num_energy_bins)
        tot_E1 = np.zeros(self.pmt_num_energy_bins)
        tot_E2 = np.zeros_like(tot_E1)
        tot_E3 = np.zeros_like(tot_E1)
        tot_E4 = np.zeros_like(tot_E1)

        for run_number, run in enumerate(self.runs):
            eng, img = run.convert_adc_to_bins(rid, sid, **kwargs)  # img = (image_hist, raw_hist)
            # image_hist = crystal segmented, raw_hist = raw anger logic (0 to 100)
            if not run_number:  # first iteration
                self.image_list[sid] = img[0]
                self.raw_image_list[sid] = img[1]
            else:
                self.image_list[sid] += img[0]
                self.raw_image_list[sid] = img[1]

            total_energy_spectra += eng[0]
            tot_E1 += eng[1]
            tot_E2 += eng[2]
            tot_E3 += eng[3]
            tot_E4 += eng[4]
        return tot_E1, tot_E2, tot_E3, tot_E4, total_energy_spectra

    def _complete_run_histograms(self, **kwargs):  # kwarg -> energy_filter
        for sid in self.system_id:
            sid_ch_ind = 4 * sid
            dyn_pmt_gain_mod = self.dyn_pmt_gains[sid_ch_ind:(sid_ch_ind + 4)]
            pmt1, pmt2, pmt3, pmt4, mod_eng = \
                self._subprocess_mod_sum_histograms(4 * self.mod_id[sid], sid,
                                                    dynamic_pmt_gains=dyn_pmt_gain_mod,
                                                    dynamic_mod_gains=self.dyn_mod_gains[sid], **kwargs)
            yield sid, [pmt1, pmt2, pmt3, pmt4, mod_eng]

    def generate_spectra(self, **kwargs):  # crystal_binning, energy_filter
        if 'filter_limits' in kwargs:
            self.filter_limits = kwargs['filter_limits']

        for sid, hists in self._complete_run_histograms(**kwargs):
            print("System ID ", str(sid), "(Module ID ", str(self.mod_id[sid]), ") Processed")
            self.pmt_histograms[(4 * sid):((4*sid) + 4)] = np.vstack(hists[:-1])  # pmt1, 2, 3, 4
            self.module_histograms[sid] = hists[-1]
        self.data_generated = True

    def display_spectra_and_image(self, mod_id=None, energy_axis=False, save_fname=None, image_name=None,
                                  pmt_legend=False, show_crystal_edges=False):
        # Display module histograms and pmt histograms
        if not self.data_generated:
            return
        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(2, 4)
        ax1 = fig.add_subplot(gs[0, :2])  # modules
        ax2 = fig.add_subplot(gs[1, :2])  # pmts
        # ax3 = fig.add_subplot(gs[:, 2:])  # projection image (old)
        ax3 = fig.add_subplot(gs[0, 2:])  # projection image
        ax4 = fig.add_subplot(gs[1, 2:])  # raw image (energy weights)

        x_mod = np.linspace(0, self.runs[0].mod_histogram_bins[-1], self.runs[0].mod_histogram_bins.size - 1)  # mods
        x_pmt = np.linspace(0, self.runs[0].pmt_histogram_bins[-1], self.runs[0].pmt_histogram_bins.size - 1)  # PMTs

        if energy_axis:
           x_mod[1:] = 2 * np.log(x_mod[1:]) - 17.

        if mod_id is not None:
            mods = np.array([mod_id])
            image = self.image_list[mod_id]
            raw_image = self.raw_image_list[mod_id]  # TODO: Added
        else:
            mods = self.system_id  # all of them
            image = self.full_image(self.image_list)
            raw_image = self.full_image(self.raw_image_list)  # TODO: Added

        # print("Mods: ", mods)
        for sid in mods:
            ax1.step(x_mod, self.module_histograms[sid], where='mid', label='mod ' + str(self.mod_id[sid]))
            for pmt_ind in np.arange(4):
                pmt_num = (4 * self.mod_id[sid] + pmt_ind)
                ax2.step(x_pmt, self.pmt_histograms[pmt_num],
                         where='mid',
                         label='m' + str(self.mod_id[sid]) + ' p' + str(pmt_ind))

        # Crystal Map
        im = ax3.imshow(image, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal')

        ax3.axis('off')
        fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax3)
        if type(image_name) is str:
            ax3.set_title(image_name)
        # Crystal Map

        # Raw Map
        im_r = ax4.imshow(raw_image, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal',
                          extent=(0, 100, 0, 100))

        fig.colorbar(im_r, fraction=0.046, pad=0.04, ax=ax4)
        ax4.set_title("Raw Weighted Map")
        if show_crystal_edges:
            for sid in mods:
                x_cuts = self.runs[0].crude_crystal_cutsX[sid]
                # y_cuts = 100 - self.runs[0].crude_crystal_cutsY[sid]  # TODO: Check this flip is needed
                y_cuts = self.runs[0].crude_crystal_cutsY[sid]  # TODO: Check if flip is needed
                ax4.vlines(x=x_cuts, ymin=0, ymax=100, colors='black', linestyles='-', lw=2)
                ax4.hlines(y=y_cuts, xmin=0, xmax=100, colors='black', linestyles='-', lw=2)
        ax4.axis('off')
        # Raw Map

        ax1.set_yscale('log')
        ax1.set_title('Module Spectrum')

        ll = np.min(np.argmax(self.module_histograms > 0, axis=1))
        ul = self.module_histograms.shape[1] - np.min(np.argmax(np.fliplr(self.module_histograms) > 0, axis=1)) - 1

        ax1.set_xlim([np.max([0, 0.9 * x_mod[ll]]), np.min([1.01 * x_mod[ul], 1.01 * x_mod[-1]])])
        ax1.legend(loc='upper right')

        pll = np.min(np.argmax(self.pmt_histograms > 0, axis=1))
        pul = self.pmt_histograms.shape[1] - np.min(np.argmax(np.fliplr(self.pmt_histograms) > 0, axis=1)) - 1

        ax2.set_xlim([np.max([0, 0.9 * x_pmt[pll]]), np.min([1.01 * x_pmt[pul], 1.01 * x_pmt[-1]])])
        ax2.set_yscale('log')
        ax2.set_title('PMT Spectrum')
        if pmt_legend:
            ax2.legend(loc='best')

        fig.tight_layout()
        if type(save_fname) is str:
            plt.savefig(save_fname, bbox_inches="tight")

        return fig, [ax1, ax2, ax3]

    @staticmethod
    def full_image(image_list):
        return np.block([image_list[col:col + 4] for col in np.arange(0, len(image_list), 4)])

    def calibrate_pmt_gains(self):  # use gaussian_filter from scipy along axis
        pass

    def calibrate_mod_gains(self, roi_center, roi_window, shape='edge', ma_sze=3):  # , shape_width=5):
        if shape not in ('edge', 'peak') or not self.data_generated:
            return

        bn = np.diff(self.runs[0].mod_histogram_bins[:2])
        cnt_ind = roi_center//bn
        w = roi_window//bn
        region_of_interest = self.module_histograms[:,  np.arange(cnt_ind-w, cnt_ind + w + 1)]
        # issues if cnt_ind -/+ w is near edges

        mode = 'nearest'

        smoothed = uniform_filter1d(1.0 * region_of_interest, ma_sze, axis=1, mode=mode)

        if shape == 'edge':
            logged = np.log(smoothed, out=np.zeros_like(smoothed), where=(smoothed != 0))

            mod_ref_pts = np.argmax(np.diff(logged, axis=1), axis=1) + cnt_ind - w
            mod_standard = np.argmin(mod_ref_pts)
            print("Old Dynamic Mod Gains: ", self.dyn_mod_gains)
            self.dyn_mod_gains = mod_ref_pts[mod_standard] / mod_ref_pts
            print("New Dynamic Mod Gains: ", self.dyn_mod_gains)

    def save_hist_and_calib(self, filename):
        np.savez(filename, dyn_pmt_gains=self.dyn_pmt_gains,
                 dyn_mod_gains=self.dyn_mod_gains,
                 pmt_histograms=self.pmt_histograms,
                 pmt_histogram_bins=self.runs[0].pmt_histogram_bins,  #
                 filter_limits=self.filter_limits,
                 # energy_filter=self.energy_filter,
                 module_histograms=self.module_histograms,
                 mod_histogram_bins=self.runs[0].mod_histogram_bins,  #
                 image_list=self.full_image(self.image_list),
                 raw_image_list=self.full_image(self.raw_image_list))

    def load_hist_and_calib(self, filename):  # TODO: Break up image_list attribute back into a list
        data = np.load(filename)
        for key, value in data.items():
            if key in ('image_list', 'raw_image_list'):
                dim_mod = value.shape[0]//4
                tmp = value.reshape(4, dim_mod, 4, dim_mod).swapaxes(1, 2).reshape(-1, dim_mod, dim_mod)
                img_list = [tmp[ind] for ind in np.arange(16)]
                setattr(self, key, img_list)
            if key in ('mod_histogram_bins', 'pmt_histogram_bins'):
                for run in self.runs:
                    setattr(run, key, value)
                continue
            setattr(self, key, value)
        self.data_generated = True  # TODO: Currently save entire image, break back up into list


class Hist1D(object):
    def __init__(self, xbins):
        self.hist, self.xbins = np.histogram([], bins=xbins)

    def fill(self, arr):
        self.hist += np.histogram(arr, bins=self.xbins)[0]

    def clear(self):
        self.hist.fill(0)

    @property
    def data(self):
        return self.xbins, self.hist.copy()


class HistImage(object):  # TODO: Added
    def __init__(self, crystal_cutsX, crystal_cutsY, raw_min=0, raw_max=100, raw_bins=101):
        self.raw_bin_edges = np.linspace(raw_min, raw_max, raw_bins)
        self.hist, self.ybins, self.xbins = \
            np.histogram2d([], [], bins=[crystal_cutsY[0].ravel(), crystal_cutsX[0].ravel()])
        self.raw_hist, _, _ = np.histogram2d([], [], bins=[self.raw_bin_edges, self.raw_bin_edges])
        self.x_edges_table = crystal_cutsX
        self.y_edges_table = crystal_cutsY
        self.current_module = 0

    def select(self, mod_id):
        self.current_module = mod_id
        self.clear()
        self.xbins = self.x_edges_table[mod_id].ravel()
        self.ybins = self.y_edges_table[mod_id].ravel()

    def fill(self, xarr, yarr):  # These get switched when entered
        self.raw_hist += np.histogram2d(yarr, xarr, bins=[self.raw_bin_edges, self.raw_bin_edges])[0][::-1]
        self.hist += np.histogram2d(yarr, xarr, bins=[self.ybins, self.xbins])[0][::-1]
        # The [::-1] index is needed to reverse what histogram does i.e. turn it back rightside up

    def clear(self):
        self.hist.fill(0)
        self.raw_hist.fill(0)

    @property
    def data(self):
        return self.xbins, self.ybins, self.hist.copy(), self.raw_hist.copy()


def load_signals(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def main_th_measurement():  # one_module_processing for outstanding issues
    base_path = '/home/justin/Desktop/Davis_Data_Backup/'
    folder = 'Wednesday/calib_in_BP_spot/OvernightTh/'
    files = ['2020-10-07-1940.h5']  # Davis data
    location = "Davis"
    filepaths = [base_path + folder + file for file in files]
    full_run = system_processing(filepaths, place=location, mod_adc_max_bin=80000, mod_adc_bin_size=150, pmt_adc_max_bin=40000)

    e_filter = [20000, 40000]
    full_run.generate_spectra(filter_limits=e_filter)

    base_save_path = '/home/justin/Desktop/images/May5/crystal_check/'
    mod_path = base_save_path + 'Xchanged_Mod'
    data_name = base_save_path + 'thor10_07_overnight'
    # TODO: Test crystal cuts
    for mod in np.arange(1) + 8:
        fig, axes = full_run.display_spectra_and_image(mod_id=mod,
                                                       save_fname=mod_path + str(mod),
                                                       show_crystal_edges=True)
        # fig, axes = full_run.display_spectra_and_image(mod_id=mod, show_crystal_edges=True)
        plt.show()
    print("Total Events: ", full_run.module_histograms.sum())

    for run in full_run.runs:
        run.h5file.close()

    # full_run.save_hist_and_calib(filename=data_name)


def main_display(steps, mods=None, area='Mid', **kwargs):   # TODO: Modify for saved data above so you can tweak edges
    if mods is None:
        mods = np.arange(16)

    base_folder = '/home/justin/Desktop/processed_data/'
    working_folder = 'uncalibrated_middle/'
    if area.lower() == 'corner':
        working_folder = 'uncalibrated_corner_b3/'
    if area.lower() == 'full':
        working_folder = 'calibrated_full/'

    ranges = ['0t1', '1t2', '2t3', '3t4', '4t5', '5t6', '6t7', '7t8', '8t9', '9t10']
    rngs = [ranges[step] for step in steps]
    run_objs = []
    location = 'Davis'

    processed_files = [base_folder + working_folder + 'step_run_' + rng + 'cm_Apr10.npz' for rng in rngs]
    base_path, data_file_lists = run_mm_steps(steps=steps)

    for pid, pfile in enumerate(processed_files):
        filepaths = [base_path + file for file in data_file_lists[pid]]
        full_run = system_processing(filepaths, place=location,
                                     mod_adc_max_bin=180000,
                                     mod_adc_bin_size=150,
                                     pmt_adc_max_bin=90000)
        full_run.load_hist_and_calib(pfile)
        run_objs.append(full_run)

        for mod in mods:
            # fig, axes = full_run.display_spectra_and_image(mod_id=mod, **kwargs)
            print("Total {m} Events: {c}".format(m=mod, c=full_run.module_histograms[mod].sum()))
            # plt.show()

        print("Total Events: ", full_run.module_histograms.sum())

        full_run.display_spectra_and_image(energy_axis=False)  # TODO: ALl Modules
        plt.show()

    for obj in run_objs:
        for run in obj.runs:
            run.h5file.close()


if __name__ == "__main__":
    main_th_measurement()

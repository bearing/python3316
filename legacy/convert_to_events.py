import tables
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from processing.calibration_values import load_calibration
# from scipy import stats  # This will be necessary to do a per-pixel gain calibration
from scipy.ndimage import uniform_filter1d
# or single pixel spectrum (scipy.binned_statistic2d)


class events_recon(object):

    def __init__(self, filepath, calib=None, place='Davis', mod_adc_max_bin=18000, mod_adc_bin_size=250,
                 pmt_adc_max_bin=18000, pmt_adc_bin_size=250):
        if calib is None:  # calib should be a dict
            calib = load_calibration(place)

        self._pmts = calib['pmts']
        self.swapped = calib['swapped']
        self._pmt_swapped = calib['swapped_pmts']

        # self.initialize = True
        # self.h5file = filepath  # load_signals(filepath)
        self.h5file = load_signals(filepath)

        # self.histogram_bins = np.arange(0, 180000, 1000)
        self.mod_histogram_bins = np.arange(0, mod_adc_max_bin + mod_adc_bin_size, mod_adc_bin_size)
        self.pmt_histogram_bins = np.arange(0, pmt_adc_max_bin + pmt_adc_bin_size, pmt_adc_bin_size)
        self.image_size = np.array([100, 100])

        self.gamma_events = 0
        self.filtered_gamma_events = 0

        self.pmt_gains = calib['pmt_gains']
        self.pmt_shifts = calib['pmt_shifts']
        self.module_gains = calib['module_gains']
        self.module_shifts = calib['module_shifts']

        # self.dynamic_pmt_gains = np.ones_like(self.pmt_gains)  # Added here, varies per run
        # self.dynamic_pmt_shifts = np.zeros_like(self.pmt_shifts)  # Added here, varies per run

        self.alphas = calib['alpha_undistort']
        self.com_bins = np.linspace(-1.0, 1.0, 101)
        self.crude_crystal_cutsX = calib['crystal_x_edges']
        self.crude_crystal_cutsY = calib['crystal_y_edges']

        self.energy_coeff_a = calib['energy_a']
        self.energy_coeff_b = calib['energy_b']

    def convert_to_bins(self, ch_start_index, mod_ref_index, dynamic_pmt_gains=np.ones(4), dynamic_mod_gains=1,
                        crystal_bin=True, filter_limits=None, energy_filter=False):
        # filter_limits is formerly what energy_filter was. energy_filter now selects type (ADC vs. Energy)
        # formerly projection
        # Ch_start index is the channel as read by the digitizer. Mod ref index is as defined globally (mod id)
        pmts = np.arange(4)
        tables = [0 for _ in np.arange(4)]

        if filter_limits is None:
            filter_limits = [0]

        if len(filter_limits) >= 2:  # MeV
            filter_limits.sort()

        for integer in pmts:
            folder = '/det' + str(int(ch_start_index + integer))
            tables[integer] = self.h5file.get_node('/', folder).EventData

        evts = tables[0].nrows
        if evts < 1:
            return None, None

        energy_array = np.zeros([4, evts])  # evts are columns

        self.gamma_events += evts

        if self.swapped[mod_ref_index]:
            pmts_ch_map = self._pmt_swapped
            # print("Swapped. Mod ref idx:", mod_ref_index, ". ch_start:", ch_start_index//4)
        else:
            pmts_ch_map = self._pmts

        # When facing the front of the detectors. This maps spatial position of PMTs to your cabling
        ul = pmts_ch_map[0][0]
        ur = pmts_ch_map[0][1]
        ll = pmts_ch_map[1][0]
        lr = pmts_ch_map[1][1]
        pmt_ch_map = np.array([ul, ur, ll, lr])

        for pmt_id, ch_id in enumerate(pmt_ch_map):
            # pmt_id = 0, 1, 2, 3
            energy_array[pmt_id] = (self.pmt_gains[mod_ref_index, pmt_id] * (tables[ch_id].col('gate2')
                                                                             - 3.0 * tables[ch_id].col('gate1')) -
                                    self.pmt_shifts[mod_ref_index, pmt_id]) * dynamic_pmt_gains[pmt_id]  # This is new

        all_sum = np.sum(energy_array, axis=0) * dynamic_mod_gains
        valid_evts = ((np.sum(energy_array, axis=0) * dynamic_mod_gains) > 0)

        valid_pmts = energy_array[:, valid_evts]
        raw_sum = all_sum[valid_evts]

        # TODO: Does this still work?
        Ex = ((valid_pmts[1] - valid_pmts[0]) + (valid_pmts[3] - valid_pmts[2])) / (1.0 * raw_sum)
        Ey = ((valid_pmts[1] - valid_pmts[3]) + (valid_pmts[0] - valid_pmts[2])) / (1.0 * raw_sum)

        if not crystal_bin:  # i.e. 100 x 100 bins per module
            img = np.histogram2d(Ex, Ey, bins=[self.com_bins, self.com_bins])[0]
            sum_e = raw_sum
            # pmt_evts_filtered = np.ones_like(sum_e)
            pmt_evts_filtered = np.full(sum_e.shape, 1, dtype=bool)
        else:
            if energy_filter:
                sum_scaled = self.energy_coeff_a[mod_ref_index] * np.log(1.0 * raw_sum) - self.energy_coeff_b[
                    mod_ref_index]
            else:
                sum_scaled = raw_sum

            s_filter = (sum_scaled > filter_limits[0])  # & (raw_sum > 0) # Is this needed?

            if len(filter_limits) is 2:
                s_filter &= (sum_scaled < filter_limits[1])

            sum_e = raw_sum[s_filter]
            pmt_evts_filtered = s_filter

            filt_Ex_scaled = (Ex[s_filter] + 1)/0.02
            filt_Ey_scaled = (Ey[s_filter] + 1)/0.02

            filt_Ex_scaled[filt_Ex_scaled > 100] = 99
            filt_Ey_scaled[filt_Ey_scaled > 100] = 99

            filt_Ex_scaled[filt_Ex_scaled < 0] = 1  # Check: Is this needed?
            filt_Ey_scaled[filt_Ey_scaled < 0] = 1

            img = np.histogram2d(filt_Ex_scaled, filt_Ey_scaled,
                                 bins=[self.crude_crystal_cutsX[mod_ref_index].ravel(),
                                       self.crude_crystal_cutsY[mod_ref_index].ravel()])[0]

        self.image_size = img.shape

        energy_hist = np.histogram(sum_e, bins=self.mod_histogram_bins)[0]
        self.filtered_gamma_events += np.sum(energy_hist)

        E1 = np.histogram(valid_pmts[0, pmt_evts_filtered], bins=self.pmt_histogram_bins)[0]  # ul
        E2 = np.histogram(valid_pmts[1, pmt_evts_filtered], bins=self.pmt_histogram_bins)[0]  # ur
        E3 = np.histogram(valid_pmts[2, pmt_evts_filtered], bins=self.pmt_histogram_bins)[0]  # ll
        E4 = np.histogram(valid_pmts[3, pmt_evts_filtered], bins=self.pmt_histogram_bins)[0]  # lr

        return [energy_hist, E1, E2, E3, E4], img

    # def select_pixel(self):  # i.e. select bin for calibration or specturm purposes
    #     pass


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
            self.mod_id = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])  # The order they were
        # plugged in by channel id
            print("Place: Davis")
        else:
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
        self.filter_limits = [0]
        self.energy_filter = True

    def _subprocess_mod_sum_histograms(self, rid, sid, **kwargs):  # raw_id (digitizer), system_id (global labeling)
        total_energy_spectra = np.zeros(self.mod_num_energy_bins)
        tot_E1 = np.zeros(self.pmt_num_energy_bins)
        tot_E2 = np.zeros_like(tot_E1)
        tot_E3 = np.zeros_like(tot_E1)
        tot_E4 = np.zeros_like(tot_E1)

        for run_number, run in enumerate(self.runs):

            eng, img = run.convert_to_bins(rid, sid, **kwargs)
            if not run_number:  # first iteration
                self.image_list[sid] = img
            else:
                self.image_list[sid] += img

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
        if 'energy_filter' in kwargs:
            self.energy_filter = kwargs['energy_filter']

        for sid, hists in self._complete_run_histograms(**kwargs):
            print("System ID ", str(sid), "(Module ID ", str(self.mod_id[sid]), ") Processed")
            self.pmt_histograms[(4 * sid):((4*sid) + 4)] = np.vstack(hists[:-1])  # pmt1, 2, 3, 4
            self.module_histograms[sid] = hists[-1]
        self.data_generated = True

    def display_spectra_and_image(self, energy_axis=False, save_fname=None, image_name=None):
        # Display module histograms and pmt histograms
        if not self.data_generated:
            return
        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(2, 4)
        ax1 = fig.add_subplot(gs[0, :2])  # modules
        ax2 = fig.add_subplot(gs[1, :2])  # pmts
        ax3 = fig.add_subplot(gs[:, 2:])  # projection image

        x_mod = np.linspace(0, self.runs[0].mod_histogram_bins[-1], self.runs[0].mod_histogram_bins.size - 1)  # mods
        x_pmt = np.linspace(0, self.runs[0].pmt_histogram_bins[-1], self.runs[0].pmt_histogram_bins.size - 1)  # PMTs

        if energy_axis:
           x_mod[1:] = 2 * np.log(x_mod[1:]) - 17.5

        for sid in self.system_id:
            ax1.step(x_mod, self.module_histograms[sid], where='mid', label='mod ' + str(self.mod_id[sid]))
            for pmt_ind in np.arange(4):
                pmt_num = (4 * self.mod_id[sid] + pmt_ind)
                ax2.step(x_pmt, self.pmt_histograms[pmt_num],
                         where='mid',
                         label='m' + str(self.mod_id[sid]) + 'p' + str(pmt_ind))

        # self.image_list[0] = np.zeros([12, 12])  # orient up/down for image
        image = self.full_image(self.image_list)

        im = ax3.imshow(image.T, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal')
        ax3.axis('off')
        fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax3)
        if type(image_name) is str:
            ax3.set_title(image_name)

        ax1.set_yscale('log')
        ax1.set_title('Module Spectrum')

        ll = np.min(np.argmax(self.module_histograms > 0, axis=1))
        ul = self.module_histograms.shape[1] - np.min(np.argmax(np.fliplr(self.module_histograms) > 0, axis=1)) - 1

        ax1.set_xlim([np.max([0, 0.9 * x_mod[ll]]), np.min([1.01 * x_mod[ul], 1.01 * x_mod[-1]])])
        # ax1.legend(loc='best')
        ax1.legend(loc='upper right')

        ax2.set_yscale('log')
        ax2.set_title('PMT Spectrum')

        fig.tight_layout()
        if type(save_fname) is str:
            plt.savefig(save_fname, bbox_inches="tight")

        return fig, [ax1, ax2, ax3]

        # plt.show()

    @staticmethod
    def full_image(image_list):
        return np.block([image_list[col:col + 4] for col in np.arange(0, len(image_list), 4)])

    def calibrate_pmt_gains(self):  # use gaussian_filter from scipy along axis
        pass

    def calibrate_mod_gains(self, roi_center, roi_window, shape='edge', ma_sze=3, shape_width=5):
        if shape not in ('edge', 'peak') or not self.data_generated:
            return

        bn = np.diff(self.runs[0].mod_histogram_bins[:2])
        cnt_ind = roi_center//bn
        w = roi_window//bn
        region_of_interest = self.module_histograms[:,  np.arange(cnt_ind-w, cnt_ind + w + 1)]
        # issues if cnt_ind -/+ w is near edges

        mode = 'nearest'

        smoothed = uniform_filter1d(1.0 * region_of_interest, ma_sze, axis=1, mode=mode)

        if shape is 'edge':
            logged = np.log(smoothed, out=np.zeros_like(smoothed), where=(smoothed != 0))

            # mod_ref_pts = np.argmax((smoothed < (np.max(smoothed, axis=1)/2)[:, np.newaxis]), axis=1)  # amplitude
            # mod_standard = np.argmin(mod_ref_pts)
            # mod_ref_pts = np.argmax(np.diff(smoothed, axis=1), axis=1) + cnt_ind - w  # slope
            # mod_standard = np.argmin(mod_ref_pts)
            mod_ref_pts = np.argmax(np.diff(logged, axis=1), axis=1) + cnt_ind - w
            mod_standard = np.argmin(mod_ref_pts)
            print("Old Dynamic Mod Gains: ", self.dyn_mod_gains)
            self.dyn_mod_gains = mod_ref_pts[mod_standard] / mod_ref_pts
            print("New Dynamic Mod Gains: ", self.dyn_mod_gains)

    def save_hist_and_calib(self, filename):  # TODO: image_list is not saving or loading correctly
        np.savez(filename, dyn_pmt_gains=self.dyn_pmt_gains,
                 dyn_mod_gains=self.dyn_mod_gains,
                 pmt_histograms=self.pmt_histograms,
                 pmt_histogram_bins=self.runs[0].pmt_histogram_bins,  #
                 filter_limits=self.filter_limits,
                 energy_filter=self.energy_filter,
                 module_histograms=self.module_histograms,
                 mod_histogram_bins=self.runs[0].mod_histogram_bins,  #
                 image_list=self.image_list)

    def load_hist_and_calib(self, filename):
        data = np.load(filename)
        for key, value in data.items():
            if key in ('mod_histogram_bins', 'pmt_histogram_bins'):
                for run in self.runs:
                    setattr(run, key, value)
                continue
            setattr(self, key, value)
        self.data_generated = True


def load_signals(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def main_th_measurement():  # one_module_processing for outstanding issues
    base_path = '/home/proton/repos/python3316/Data/'
    files = ['2020-10-31-1704.h5']
    location = "Berkeley"
    filepaths = [base_path + file for file in files]
    full_run = system_processing(filepaths, place=location, mod_adc_max_bin=80000)

    # e_filter = [1]
    # full_run.generate_spectra(filter_limits=e_filter, energy_filter=True, crystal_bin=True)

    e_filter = [20000, 36000]  # Feb 15
    full_run.generate_spectra(filter_limits=e_filter, crystal_bin=True)
    full_run.calibrate_mod_gains(29000, 4000, ma_sze=1)
    full_run.generate_spectra(filter_limits=e_filter, crystal_bin=True)

    # fig, axes = full_run.display_spectra_and_image(save_fname="th_flood_1031_feb_15")  # to allow for changing of axes
    fig, axes = full_run.display_spectra_and_image()
    print("Total Events: ", full_run.module_histograms.sum())
    plt.show()

    for run in full_run.runs:
        run.h5file.close()
    # full_run.save_hist_and_calib(filename="th_flood_1031_feb_15")


def main_step_measurement():  # one_module_processing for outstanding issues
    base_path = '/home/proton/repos/python3316/Data/'
    files56 = ['2020-10-07-1210.h5', '2020-10-07-1219.h5', '2020-10-07-1221.h5', '2020-10-07-1222.h5',
               '2020-10-07-1224.h5', '2020-10-07-1225.h5', '2020-10-07-1226.h5', '2020-10-07-1228.h5',
               '2020-10-07-1229.h5', '2020-10-07-1230.h5']  # step measurement 50-59, 5-6 cm

    # list_of_files = [files12, files23, files34, files45, files56, files67, files78, files89, files9]
    list_of_files = [files56]

    for start in np.arange(len(list_of_files)):
        step = start + 1  # start is at 0 but first run is at 1 cm
        filepaths = [base_path + file for file in list_of_files[start]]

        save_fname = '/home/proton/repos/python3316/processing/images/step_run_' + str(step) + "t" \
                     + str(step + 1) + 'cm_Feb10'

        plot_name = 'Position ' + str(step) + '-' + str(step + 1) + ' cm'

        full_run = system_processing(filepaths, mod_adc_max_bin=160000, pmt_adc_max_bin=80000)
        e_filter = [20000, 100000]
        full_run.generate_spectra(filter_limits=e_filter)
        # full_run.display_spectra_and_image(save_fname=save_fname, image_name=plot_name)
        full_run.display_spectra_and_image()
        plt.show()
        for run in full_run.runs:
            run.h5file.close()

        # full_run.save_hist_and_calib(filename=save_fname)


def fix_loading_issues():
    pass


if __name__ == "__main__":
    # TODO: Transpose ALL histograms
    # main()
    # main_th_measurement()
    main_step_measurement()

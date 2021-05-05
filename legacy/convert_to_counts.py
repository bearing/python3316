import tables
import io
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
# from processing.single_module_processing import events_recon, load_signals
# from processing.one_module_processing import events_recon as per
from processing.calibration_values_m5 import load_calibration

# redo one_module_processing and system projection in one file


class events_recon(object):

    def __init__(self, filepath, calib=None, place='Davis'):
        if calib is None:  # calib should be a dict
            calib = load_calibration(place)

        self._pmts = calib['pmts']
        self.swapped = calib['swapped']
        self._pmt_swapped = calib['swapped_pmts']

        # self.initialize = True
        # self.h5file = filepath  # load_signals(filepath)
        self.h5file = load_signals(filepath)

        # self.histogram_bins = np.arange(0, 180000, 1000)
        self.histogram_bins = np.arange(0, 180000, 250)
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

    # @property
    # def h5file(self):
    #    return self._h5file

    # @h5file.setter
    # def h5file(self, file):
    #     if not self.initialize:
    #         self._h5file.close()
    #     else:
    #         self.initialize = False
    #    self._h5file = load_signals(file)

    def convert_to_bins(self, ch_start_index, mod_ref_index, dynamic_pmt_gains=np.ones(4), dynamic_mod_gains=1,
                        crystal_bin=True, energy_filter=None):
        # formerly projection
        # Ch_start index is the channel as read by the digitizer. Mod ref index is as defined globally (mod id)
        pmts = np.arange(4)
        tables = [0 for _ in np.arange(4)]

        if energy_filter is None:
            energy_filter = [0]

        if len(energy_filter) is 2:  # MeV
            energy_filter.sort()

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
        # TODO (2/9): using pmt_id not channel id. Is this correct?
        all_sum = np.sum(energy_array, axis=0) * dynamic_mod_gains
        valid_evts = ((np.sum(energy_array, axis=0) * dynamic_mod_gains) > 0)  # Can go negative with muon pulse shapes
        valid_pmts = energy_array[:, valid_evts]
        raw_sum = all_sum[valid_evts]

        Ex = ((energy_array[1, valid_evts] - energy_array[0, valid_evts]) +
              (energy_array[3, valid_evts]-energy_array[2, valid_evts]))/(1.0 * raw_sum)
        Ey = ((energy_array[1, valid_evts] - energy_array[3, valid_evts]) +
              (energy_array[0, valid_evts]-energy_array[2, valid_evts]))/(1.0 * raw_sum)
        # TODO: Change this summation using valid_pmts instead
        # Ex = ((energy_array[1] - energy_array[0]) + (energy_array[3] - energy_array[2])) / (1.0 * raw_sum)
        # Ey = ((energy_array[1] - energy_array[3]) + (energy_array[0] - energy_array[2])) / (1.0 * raw_sum)

        if not crystal_bin:  # i.e. 100 x 100 bins per module
            img = np.histogram2d(Ex, Ey, bins=[self.com_bins, self.com_bins])[0]
            sum_e = raw_sum
            pmt_evts_filtered = np.ones_like(sum_e)
            # self.orig_hist_size = np.array([self.com_bins.size, self.com_bins.size])  # image size, really
        else:
            # TODO: Energy calibration should be done here or loaded based on features in plot
            sum_mev = self.energy_coeff_a[mod_ref_index] * np.log(1.0 * raw_sum) - self.energy_coeff_b[
                mod_ref_index]

            e_filter = (sum_mev > energy_filter[0])  # & (raw_sum > 0) # Is this needed?

            if len(energy_filter) is 2:
                e_filter &= (sum_mev < energy_filter[1])

            sum_e = raw_sum[e_filter]
            pmt_evts_filtered = e_filter

            filt_Ex_scaled = (Ex[e_filter] + 1)/0.02
            filt_Ey_scaled = (Ey[e_filter] + 1)/0.02

            filt_Ex_scaled[filt_Ex_scaled > 100] = 99
            filt_Ey_scaled[filt_Ey_scaled > 100] = 99

            filt_Ex_scaled[filt_Ex_scaled < 0] = 1  # Check: Is this needed?
            filt_Ey_scaled[filt_Ey_scaled < 0] = 1

            img = np.histogram2d(filt_Ex_scaled, filt_Ey_scaled,
                                 bins=[self.crude_crystal_cutsX[mod_ref_index].ravel(),
                                       self.crude_crystal_cutsY[mod_ref_index].ravel()])[0]

        self.image_size = img.shape

        energy_hist = np.histogram(sum_e, bins=self.histogram_bins)[0]  # or raw_sum?
        self.filtered_gamma_events += np.sum(energy_hist)

        # E1 = np.histogram(energy_array[0], bins=self.histogram_bins)[0]   # ul
        # E2 = np.histogram(energy_array[1], bins=self.histogram_bins)[0]   # ur
        # E3 = np.histogram(energy_array[2], bins=self.histogram_bins)[0]   # ll
        # E4 = np.histogram(energy_array[3], bins=self.histogram_bins)[0]   # lr

        E1 = np.histogram(valid_pmts[0, pmt_evts_filtered], bins=self.histogram_bins)[0]  # ul
        E2 = np.histogram(valid_pmts[1, pmt_evts_filtered], bins=self.histogram_bins)[0]  # ur
        E3 = np.histogram(valid_pmts[2, pmt_evts_filtered], bins=self.histogram_bins)[0]  # ll
        E4 = np.histogram(valid_pmts[3, pmt_evts_filtered], bins=self.histogram_bins)[0]  # lr

        return [energy_hist, E1, E2, E3, E4], img


class system_processing(object):

    def __init__(self, filepaths, place="Davis"):
        if type(filepaths) == str:
            files = [filepaths]
        else:
            files = filepaths

        self.runs = []
        for file in files:
            self.runs.append(events_recon(file, place=place))

        self.system_id = np.arange(16)  # when facing front of detectors, upper left to upper right, then down
        if place == "Davis":
            self.mod_id = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])  # The order they were
        # plugged in by channel id
            print("Place: Davis")
        else:
            self.mod_id = np.arange(16)
            print("Place: Berkeley")

        self.num_energy_bins = self.runs[0].histogram_bins.size-1
        self.pmt_histograms = np.zeros([64, self.num_energy_bins])
        self.module_histograms = np.zeros([16, self.num_energy_bins])
        self.dyn_pmt_gains = np.ones(64)
        self.dyn_mod_gains = np.ones(16)

        self.data_generated = False
        self.image_list = [0 for _ in np.arange(16)]  # list of module images by SID
        self.energy_filter_bins = [0]

    def _subprocess_mod_sum_histograms(self, rid, sid, **kwargs):  # raw_id (digitizer), system_id (global labeling)
        total_energy_spectra = np.zeros(self.num_energy_bins)
        tot_E1 = np.zeros_like(total_energy_spectra)
        tot_E2 = np.zeros_like(total_energy_spectra)
        tot_E3 = np.zeros_like(total_energy_spectra)
        tot_E4 = np.zeros_like(total_energy_spectra)

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
            # print("Key Args: ", kwargs)
            pmt1, pmt2, pmt3, pmt4, mod_eng = \
                self._subprocess_mod_sum_histograms(4 * self.mod_id[sid], sid,
                                                    dynamic_pmt_gains=dyn_pmt_gain_mod,
                                                    dynamic_mod_gains=self.dyn_mod_gains[sid], **kwargs)
            yield sid, [pmt1, pmt2, pmt3, pmt4, mod_eng]
        # kwargs = dynamic_pmt_gains (size 4) <- self.dyn_pmt_gains (size 64)

    def generate_spectra(self, **kwargs):  # crystal_binning, energy_filter
        if 'energy_filter' in kwargs:  # TODO: energy or ADC bins?
            self.energy_filter_bins = kwargs['energy_filter']

        for sid, hists in self._complete_run_histograms(**kwargs):
            print("System ID ", str(sid), "(Module ID ", str(self.mod_id[sid]), ") Processed")
            self.pmt_histograms[(4 * sid):((4*sid) + 4)] = np.vstack(hists[:-1])  # pmt1, 2, 3, 4
            self.module_histograms[sid] = hists[-1]
        self.data_generated = True

    def display_spectra_and_image(self, energy_axis=True, save_fname=None, image_name=None):
        # Display module histograms and pmt histograms
        if not self.data_generated:
            return
        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(2, 4)
        ax1 = fig.add_subplot(gs[0, :2])  # modules
        ax2 = fig.add_subplot(gs[1, :2])  # pmts
        ax3 = fig.add_subplot(gs[:, 2:])  # projection image

        x_adc = np.linspace(0, self.runs[0].histogram_bins[-1], self.runs[0].histogram_bins.size - 1)
        x_e = np.copy(x_adc)

        # if energy_axis:
        #    x_e[1:] = 2 * np.log(x_e[1:]) - 17.5

        for sid in self.system_id:
            ax1.step(x_e, self.module_histograms[sid], where='mid', label='mod ' + str(self.mod_id[sid]))
            for pmt_ind in np.arange(4):
                pmt_num = (4 * self.mod_id[sid] + pmt_ind)
                ax2.step(x_adc, self.pmt_histograms[pmt_num],
                         where='mid',
                         label='m' + str(self.mod_id[sid]) + 'p' + str(pmt_ind))

        # image = [self.image_list[col:col+4] for col in np.arange(0, len(self.image_list), 4)]

        image = self.full_image(self.image_list)

        im = ax3.imshow(image.T, cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')
        ax3.axis('off')
        fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax3)
        if type(image_name) is str:
            ax3.set_title(image_name)

        ax1.set_yscale('log')
        ax1.set_title('Module Spectrum')
        ax1.set_xlim([1, 1.01 * np.max(x_e)])
        # ax1.legend(loc='best')
        ax1.legend(loc='upper right')

        ax2.set_yscale('log')
        ax2.set_title('PMT Spectrum')

        fig.tight_layout()
        if type(save_fname) is str:
            plt.savefig(save_fname, bbox_inches="tight")

        plt.show()

    @staticmethod
    def full_image(image_list):
        return np.block([image_list[col:col + 4] for col in np.arange(0, len(image_list), 4)])

    def calibrate_pmt_gains(self):  # use gaussian_filter from scipy along axis
        pass

    def calibrate_mod_gains(self):
        pass

    def save_hist_and_calib(self, filename):  # TODO: image_list is not saving or loading correctly
        np.savez(filename, dyn_pmt_gains=self.dyn_pmt_gains,
                 dyn_mod_gains=self.dyn_mod_gains,
                 pmt_histograms=self.pmt_histograms,
                 energy_filter_bins=self.energy_filter_bins,
                 module_histograms=self.module_histograms,
                 image_list=self.image_list)

    def load_hist_and_calib(self, filename):
        data = np.load(filename)
        for key, value in data.items():
            setattr(self, key, value)
        self.data_generated = True
        # self.dyn_mod_gains = data['dyn_mod_gains']


def load_signals(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def load_data(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def main_th_measurement():
    base_path = '/home/proton/repos/python3316/Data/'
    files = ['2020-10-31-1704.h5']
    location = "Berkeley"
    filepaths = [base_path + file for file in files]
    full_run = system_processing(filepaths, place=location)

    e_filter = [1, 3.4]
    full_run.generate_spectra(energy_filter=e_filter, crystal_bin=True)

    # full_run.load_hist_and_calib(filename="th_measurement_1031.npz")
    # full_run.display_spectra_and_image(save_fname="th_measurement_1031.npz")
    full_run.display_spectra_and_image()
    for run in full_run.runs:
        run.h5file.close()
    # full_run.save_hist_and_calib(filename="th_flood_1031")


def test_image_loader():
    base_path = '/home/proton/repos/python3316/Data/'
    files = ['2020-10-31-1704.h5']
    location = "Berkeley"
    filepaths = [base_path + file for file in files]
    full_run = system_processing(filepaths, place=location)
    # full_run.generate_spectra(energy_filter=[1, 3.2], crystal_bin=True)
    full_run.load_hist_and_calib(filename="th_measurement_1031.npz")
    full_run.display_spectra_and_image()
    for run in full_run.runs:
        run.h5file.close()


def main():
    base_path = '/home/proton/repos/python3316/Data/'

    files12 = ['2020-10-07-1111.h5', '2020-10-07-1112.h5']  # step measurement 12-21, 1-2 cm

    filepaths = [base_path + file for file in files12]
    full_run = system_processing(filepaths)

    # full_run.generate_spectra(energy_filter=[3.5, 7], crystal_bin=True)
    full_run.generate_spectra(energy_filter=[0, 4], crystal_bin=True)
    print("Done!")
    full_run.display_spectra_and_image()
    # full_run.display_spectra_and_image(energy_filter=[3.5, 7])
    for run in full_run.runs:
        run.h5file.close()


def main_small_steps():
    base_path = '/home/proton/repos/python3316/Data/'

    files12 = ['2020-10-07-1111.h5', '2020-10-07-1112.h5', '2020-10-07-1114.h5', '2020-10-07-1116.h5',
             '2020-10-07-1117.h5', '2020-10-07-1119.h5', '2020-10-07-1120.h5', '2020-10-07-1121.h5',
             '2020-10-07-1123.h5', '2020-10-07-1127.h5']  # step measurement 12-21, 1-2 cm

    files56 = ['2020-10-07-1210.h5', '2020-10-07-1219.h5', '2020-10-07-1221.h5', '2020-10-07-1222.h5',
             '2020-10-07-1224.h5', '2020-10-07-1225.h5', '2020-10-07-1226.h5', '2020-10-07-1228.h5',
             '2020-10-07-1229.h5', '2020-10-07-1230.h5']  # step measurement 50-59, 5-6 cm

    files9 = ['2020-10-07-1322.h5', '2020-10-07-1324.h5', '2020-10-07-1327.h5', '2020-10-07-1329.h5',
             '2020-10-07-1331.h5', '2020-10-07-1333.h5', '2020-10-07-1334.h5', '2020-10-07-1337.h5',
             '2020-10-07-1340.h5', '2020-10-07-1342.h5', '2020-10-07-1344.h5']  # step measurement 90-100, 9-10 cm

    files23 = ['2020-10-07-1129.h5', '2020-10-07-1130.h5', '2020-10-07-1132.h5', '2020-10-07-1133.h5',
             '2020-10-07-1134.h5', '2020-10-07-1136.h5', '2020-10-07-1137.h5', '2020-10-07-1139.h5',
             '2020-10-07-1140.h5']  # step measurement 22-30, 2-3 cm

    files34 = ['2020-10-07-1142.h5', '2020-10-07-1143.h5', '2020-10-07-1145.h5', '2020-10-07-1146.h5',
             '2020-10-07-1148.h5', '2020-10-07-1150.h5', '2020-10-07-1151.h5', '2020-10-07-1153.h5',
             '2020-10-07-1154.h5', '2020-10-07-1154.h5']  # step measurement 31-40, 3-4 cm

    files45 = ['2020-10-07-1157.h5', '2020-10-07-1158.h5', '2020-10-07-1200.h5', '2020-10-07-1201.h5',
             '2020-10-07-1203.h5', '2020-10-07-1204.h5', '2020-10-07-1206.h5', '2020-10-07-1207.h5',
             '2020-10-07-1209.h5']  # step measurement 41-49, 4-5 cm

    files67 = ['2020-10-07-1232.h5', '2020-10-07-1233.h5', '2020-10-07-1234.h5', '2020-10-07-1236.h5',
             '2020-10-07-1237.h5', '2020-10-07-1238.h5', '2020-10-07-1240.h5', '2020-10-07-1241.h5',
             '2020-10-07-1243.h5', '2020-10-07-1244.h5']  # step 60 - 69, 6 to 7 cm

    files78 = ['2020-10-07-1245.h5', '2020-10-07-1247.h5', '2020-10-07-1248.h5', '2020-10-07-1250.h5',
               '2020-10-07-1251.h5', '2020-10-07-1252.h5', '2020-10-07-1254.h5', '2020-10-07-1255.h5',
               '2020-10-07-1257.h5', '2020-10-07-1258.h5']  # step 70 - 79, 7 to 8 cm

    files89 = ['2020-10-07-1300.h5', '2020-10-07-1301.h5', '2020-10-07-1303.h5', '2020-10-07-1304.h5',
               '2020-10-07-1305.h5', '2020-10-07-1307.h5', '2020-10-07-1309.h5', '2020-10-07-1310.h5',
               '2020-10-07-1313.h5', '2020-10-07-1314.h5']  # step 80 - 89, 8 to 9 cm

    list_of_files = [files12, files23, files34, files45, files56, files67, files78, files89, files9]

    for start in np.arange(len(list_of_files)):
        step = start + 1  # start is at 0 but first run is at 1 cm
        filepaths = [base_path + file for file in list_of_files[start]]

        save_fname = '/home/proton/repos/python3316/processing/images/step_run_' + str(step) + "t" \
                     + str(step + 1) + 'cm_Feb10'

        plot_name = 'Position ' + str(step) + '-' + str(step + 1) + ' cm'

        full_run = system_processing(filepaths)
        full_run.generate_spectra(energy_filter=[3.5, 8])
        full_run.display_spectra_and_image(save_fname=save_fname, image_name=plot_name)
        for run in full_run.runs:
            run.h5file.close()

        full_run.save_hist_and_calib(filename=save_fname)


if __name__ == "__main__":
    # main()
    main_th_measurement()
    # main_small_steps()


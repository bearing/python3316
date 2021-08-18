import tables
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from final.calibration_values_final import load_calibration
from scipy.ndimage import uniform_filter1d
from scipy.stats import binned_statistic_2d
from processing.file_lists import run_mm_steps


class events_recon(object):
    evt_read_limit = 2 * 10 ** 4  # Number of events to read at a time # TODO: Convert to memory size using col itemsze
    pixels = 12

    def __init__(self, filepath, calib=None, place='Davis', mod_adc_max_bin=18000, mod_adc_bin_size=250,
                 pmt_adc_max_bin=18000, pmt_adc_bin_size=250, repetition_time=100):
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

        self.period = repetition_time  # Necessary for time_modulo

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
                        filter_limits=None, time_modulo=None, masks=None):  # this is version 2
        """Mask must be an array of linearized indices or a list of 4 linearized indices (PMT weighting).
        Time_modulo must be integers of timestamps modulo some value"""
        pmts = np.arange(4)
        tables = [0 for _ in pmts]

        # if time_modulo is None:
        #     time_modulo = np.arange(100)

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
                # TODO: Add Time filter here
                start = read_block_idx * self.evt_read_limit
                end = np.min([start + self.evt_read_limit, total_evts])
                energy_array[pmt_id, :(end-start)] = (self.pmt_gains[mod_ref_index, pmt_id] *
                                                      (tables[ch_id].cols.gate2[start:end] -
                                                       3.0 * tables[ch_id].cols.gate1[start:end]) -
                                                      self.pmt_shifts[mod_ref_index, pmt_id]) * dynamic_pmt_gains[pmt_id]

            if time_modulo is not None:
                t_filter = np.isin(tables[0].cols.timestamp[start:end] % 100, time_modulo)
            else:
                t_filter = np.ones(end-start, dtype=bool)

            all_sum = (np.sum(energy_array[:, :(end-start)], axis=0) * dynamic_mod_gains)

            e_bound_filter = (all_sum > filter_limits[0])
            if len(filter_limits) == 2:
                e_bound_filter &= (all_sum < filter_limits[1])

            sum_e = all_sum[e_bound_filter & t_filter]
            # print("e_bound filter shaep: ", e_bound_filter.shape)
            # print("t_filter shape: ", t_filter.shape)
            valid_pmts = energy_array[:, :(end - start)][:, e_bound_filter & t_filter]

            Ex = ((valid_pmts[1] - valid_pmts[0]) + (valid_pmts[3] - valid_pmts[2])) / (1.0 * sum_e)
            Ey = ((valid_pmts[1] - valid_pmts[3]) + (valid_pmts[0] - valid_pmts[2])) / (1.0 * sum_e)

            Ex_scaled = (Ex + 1) / 0.02
            Ey_scaled = (Ey + 1) / 0.02

            Ex_scaled[Ex_scaled > 100] = 99.
            Ey_scaled[Ey_scaled > 100] = 99.

            Ex_scaled[Ex_scaled < 0] = 1.
            Ey_scaled[Ey_scaled < 0] = 1.

            if masks is None:
                self.mod_image.fill(Ex_scaled, Ey_scaled)
                self.module_energy_histogram.fill(sum_e)

                self.E1.fill(valid_pmts[0])  # ul
                self.E2.fill(valid_pmts[1])  # ur
                self.E3.fill(valid_pmts[2])  # ll
                self.E4.fill(valid_pmts[3])  # lr
            else:
                if isinstance(masks, list):
                    self.weighted_masks_events(Ex_scaled, Ey_scaled, sum_e, valid_pmts, masks)
                else:
                    self.single_mask_events(Ex_scaled, Ey_scaled, sum_e, valid_pmts, masks)

            read_block_idx += 1

        self.filtered_gamma_events += np.sum(self.module_energy_histogram.hist)
        histograms = [self.module_energy_histogram.data[1], self.E1.data[1], self.E2.data[1], self.E3.data[1], self.E4.data[1]]
        # plt.imshow(self.mod_image.data[2], cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')
        # plt.show()
        return histograms, self.mod_image.data[2]

    def single_mask_events(self, ex, ey, sum_e, valid_pmts, mask):  # NOTE: Problems likely here
        """Mask must be linearized indices that of the  upside down pixel map"""
        xbins, ybins, _ = self.mod_image.data

        ret = binned_statistic_2d(ey, ex, 0, 'count', bins=[ybins, xbins])  # histogram returns along row than col
        self.mod_image.masked_add(ret.statistic, mask, self.pixels)  # MASK and RET.STATISTIC will be upside down

        pxl_filter = np.isin(ret.binnumber, mask)
        self.module_energy_histogram.fill(sum_e[pxl_filter])
        self.E1.fill(valid_pmts[0, pxl_filter])  # ul
        self.E2.fill(valid_pmts[1, pxl_filter])  # ur
        self.E3.fill(valid_pmts[2, pxl_filter])  # ll
        self.E4.fill(valid_pmts[3, pxl_filter])  # lr

    def weighted_masks_events(self, ex, ey, sum_e, valid_pmts, masks):
        """Masks must be a list of linearized indices that of the  upside down pixel maps"""
        xbins, ybins, _ = self.mod_image.data
        ret = binned_statistic_2d(ey, ex, 0, 'count', bins=[ybins, xbins])

        for idx, (pmt_hist, mask) in enumerate(zip(self.pmt_histograms, masks)):
            #  pmt is an object and masks is now a single mask from the list
            self.mod_image.masked_add(ret.statistic, mask, self.pixels)

            pxl_filter = np.isin(ret.binnumber, mask)
            self.module_energy_histogram.fill(sum_e[pxl_filter])
            pmt_hist.fill(valid_pmts[idx, pxl_filter])


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
        self.filter_limits = [0]

    def _subprocess_mod_sum_histograms(self, rid, sid, **kwargs):  # raw_id (digitizer), system_id (global labeling)
        total_energy_spectra = np.zeros(self.mod_num_energy_bins)
        tot_E1 = np.zeros(self.pmt_num_energy_bins)
        tot_E2 = np.zeros_like(tot_E1)
        tot_E3 = np.zeros_like(tot_E1)
        tot_E4 = np.zeros_like(tot_E1)

        for run_number, run in enumerate(self.runs):
            eng, img = run.convert_adc_to_bins(rid, sid, **kwargs)
            if not run_number:  # first iteration
                # print("SID ", sid, " img.sum() : ", img.sum())
                # print("SID ", sid, " img.mean() : ", img.mean())
                # plt.imshow(img, cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')
                # plt.show()
                self.image_list[sid] = img
            else:
                self.image_list[sid] += img

            total_energy_spectra += eng[0]
            tot_E1 += eng[1]
            tot_E2 += eng[2]
            tot_E3 += eng[3]
            tot_E4 += eng[4]
        return tot_E1, tot_E2, tot_E3, tot_E4, total_energy_spectra

    def _complete_run_histograms(self, choose_mods=np.arange(16), **kwargs):   # kwarg -> energy_filter
        for sid in np.intersect1d(choose_mods, self.system_id):  # for sid in self.system_id:
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
                                  pmt_legend=False):
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
           x_mod[1:] = 2 * np.log(x_mod[1:]) - 17.

        if mod_id is not None:
            mods = np.array([mod_id])
            image = self.image_list[mod_id]
        else:
            mods = self.system_id  # all of them
            image = self.full_image(self.image_list)

        # print("Mods: ", mods)
        for sid in mods:
            # ax1.step(x_mod, self.module_histograms[sid], where='mid', label='mod ' + str(self.mod_id[sid]))  # mod id
            ax1.step(x_mod, self.module_histograms[sid], where='mid', label='mod ' + str(sid))  # system id
            for pmt_ind in np.arange(4):
                pmt_num = (4 * self.mod_id[sid] + pmt_ind)
                ax2.step(x_pmt, self.pmt_histograms[4 * sid + pmt_ind],  # TODO: Needed?
                # ax2.step(x_pmt, self.pmt_histograms[pmt_num],
                         where='mid',
                         label='m' + str(self.mod_id[sid]) + ' p' + str(pmt_ind))  # NOTE: This not not the same as SID

        im = ax3.imshow(image, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal')
        ax3.axis('off')
        fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax3)
        if type(image_name) is str:
            ax3.set_title(image_name, fontsize=16)

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

    def save_hist_and_calib(self, filename):
        np.savez(filename, dyn_pmt_gains=self.dyn_pmt_gains,
                 dyn_mod_gains=self.dyn_mod_gains,
                 pmt_histograms=self.pmt_histograms,
                 pmt_histogram_bins=self.runs[0].pmt_histogram_bins,  #
                 filter_limits=self.filter_limits,
                 # energy_filter=self.energy_filter,
                 module_histograms=self.module_histograms,
                 mod_histogram_bins=self.runs[0].mod_histogram_bins,  #
                 image_list=self.full_image(self.image_list))

    def load_hist_and_calib(self, filename):
        data = np.load(filename)
        for key, value in data.items():
            if key in ('image_list'):
                tmp = value.reshape(value.shape[0]//12, 12, value.shape[1]//12, 12).swapaxes(1, 2).reshape(-1, 12, 12)
                img_list = [tmp[ind] for ind in np.arange(16)]
                setattr(self, key, img_list)
                continue
            if key in ('mod_histogram_bins', 'pmt_histogram_bins'):
                for run in self.runs:
                    setattr(run, key, value)
                continue
            setattr(self, key, value)
        self.data_generated = True


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


class HistImage(object):
    def __init__(self, crystal_cutsX, crystal_cutsY):
        self.hist, self.ybins, self.xbins = \
            np.histogram2d([], [], bins=[crystal_cutsY[0].ravel(), crystal_cutsX[0].ravel()])
        self.x_edges_table = crystal_cutsX
        self.y_edges_table = crystal_cutsY
        self.current_module = 0

    def select(self, mod_id):
        self.current_module = mod_id
        self.clear()
        self.xbins = self.x_edges_table[mod_id].ravel()
        self.ybins = self.y_edges_table[mod_id].ravel()

    def fill(self, xarr, yarr):  # These get switched when entered
        self.hist += np.histogram2d(yarr, xarr, bins=[self.ybins, self.xbins])[0][::-1]  # .T
        # The [::-1] index is needed to reverse what histogram does i.e. turn it back rightside up

    def clear(self):
        self.hist.fill(0)

    def masked_add(self, counts, mask, pixels):
        unravel_mask = np.zeros([pixels + 2, pixels + 2])
        unravel_mask[np.unravel_index(mask, unravel_mask.shape)] = 1
        self.hist += (counts * unravel_mask[1:-1, 1:-1])[::-1]  # Must revert to right side up

    @property
    def data(self):
        return self.xbins, self.ybins, self.hist.copy()


def load_signals(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def create_mask(type='corners', buffer=3, single_pxls=np.array([0, 0]), pixels=12):
    # The purpose of this is for use with multiplying with ret.binnumber
    if type == 'corners':
        mask = np.zeros([pixels, pixels])

        pmt_id = np.arange(4)  # pmt_ch_map = np.array([ul, ur, ll, lr])
        row = pmt_id // 2  # array([0, 0, 1, 1])
        col = pmt_id % 2  # array([0, 1, 0, 1])

        masks = []
        for r, c in zip(row, col):
            mask.fill(0)
            start_row = r * (pixels-buffer)
            start_col = c * (pixels-buffer)
            mask[start_row:(start_row+buffer), start_col:(start_col+buffer)] = 1
            masks.append(np.ravel_multi_index(np.where(np.pad(mask[::-1], 1)), np.asarray(mask.shape) + 2))

    else:  # single points
        mask = np.zeros([pixels, pixels])
        mask[single_pxls[..., 0], single_pxls[..., 1]] = 1  # Need to flip this
        masks = np.ravel_multi_index(np.where(np.pad(mask[::-1], 1)), np.asarray(mask.shape)+2)

    return masks


def time_select_indices(percent_min, percent_max):
    """The purpose of this function is to create time_modulo sets. Percent_min and max are of rf"""
    period = 100
    possible_sample_indices = np.arange(period)  # these are in samples NOT nanoseconds
    sampling_freq = 4  # ns (250 MHz)
    rf = (1/22.5 * 1000)  # 44.444 ns (22.5 MHz)
    t_min = percent_min * rf
    t_max = percent_max * rf
    rel_time = (sampling_freq * possible_sample_indices) % rf
    return (t_min, t_max), possible_sample_indices[(rel_time > t_min) & (rel_time < t_max)]


def process_projection():  # one_module_processing for outstanding issues

    # === 6 cm thick ===
    base_path = '/home/justin/Desktop/Davis_Data_Backup/Wednesday/Second_20_minutes_6_cm_thick/'
    files = ['2020-10-07-1457.h5']  # TODO: Need to find relative to plastic timestamps to unify more than 1

    # 50 cm location
    # base_path = '/home/justin/Desktop/Davis_Data_Backup/Wednesday/thick_10cmFoV_mmsteps/'
    # files = ['2020-10-07-1210.h5']

    location = "Davis"  # was Berkeley (Davis, Berkeley, Fix)
    filepaths = [base_path + file for file in files]
    full_run = system_processing(filepaths, place=location, mod_adc_max_bin=100000,
                                 mod_adc_bin_size=150, pmt_adc_max_bin=80000)
    # full_run = system_processing(base_path + mm_run_file, place=location, mod_adc_max_bin=140000,  # 100000 dflt
    #                             mod_adc_bin_size=150,  # 150 dflt
    #                             pmt_adc_max_bin=100000)  # 80000 dflt

    # === Rough Calibration === # TODO: Delete for uncalibrated
    mod_calib = np.array([4.8, 5.06, 4.77, 4.73,  # beam on
                          4.69, 5.03, 5.02, 4.82,
                          5.34, 4.78, 5.16, 4.97,
                          4.38, 4.24, 4.85, 4.45])
    calib_beam_factor = 10.5 / 11  # 10.5/11  # This accounts for average gain shift relative to beam off (Th-228 data)
    rel_calib_factor = np.array([1, 1, 1, 1,
                                 0.98, 1, 1, 1.025,
                                 1, 1, 1, 1.03,
                                 0.97, 0.94, 0.955, 1])  # relative to SID 9

    full_run.dyn_mod_gains = mod_calib.mean() / mod_calib * calib_beam_factor * rel_calib_factor
    # === Rough Calibration === # TODO: Delete for uncalibrated

    e_filter = [30000, 55000]  # Feb 15, March 16 [20000, 80000], Apr 12 [30000, 55000] i.e. C, SE, and DE

    num_tbins = 10
    # for percent in 1/num_tbins * np.arange(num_tbins):
    # base_folder = '/home/justin/Desktop/images/Apr19/6cm_time_binned/'  # uncomment to save (1 of 4)
    # sub_folder = 'processed_data/'  # uncomment to save (2 of 4)
    for id, percent in enumerate(1/num_tbins * np.arange(num_tbins)):
        if not id == 0:
            continue
        (tmin, tmax), time_pts = time_select_indices(percent, percent+0.1)
        full_run.generate_spectra(filter_limits=e_filter, time_modulo=time_pts)

        fig, axes = full_run.display_spectra_and_image(image_name='{:.2f} to {:.2f} ns'.format(tmin, tmax))
                                                       # ,save_fname=save_name)
        print("Total Events: ", full_run.module_histograms.sum())

        # save_name = base_folder + str(id)  # uncomment to save (3 of 4)
        # full_run.save_hist_and_calib(base_folder + sub_folder + str(id))  # uncomment to save (4 of 4)
        #if id == 0:
        plt.show()

    # fig, axes = full_run.display_spectra_and_image(image_name='Hi Justin')
    # print("Total Events: ", full_run.module_histograms.sum())

    for run in full_run.runs:
        run.h5file.close()

    # plt.show()

    # b_path = '/home/justin/Desktop/images/recon/'
    # sub_path = 'thick07/'
    # f_name = '6cm_filt'  # filt = [30000, 55000]
    # full_run.save_hist_and_calib(filename=b_path + sub_path + f_name)

    # scin_folder = '/det' + str(64)
    # self.scin_evts = self.h5file.get_node('/', scin_folder).EventData
    # self.scin_waveforms = self.h5file.get_node('/', scin_folder).raw_data


def process_projection_mod(mod):  # one_module_processing for outstanding issues

    # === 6 cm thick ===
    base_path = '/home/justin/Desktop/Davis_Data_Backup/Wednesday/Second_20_minutes_6_cm_thick/'
    files = ['2020-10-07-1457.h5']  # TODO: Need to find relative to plastic timestamps to unify more than 1

    # 50 cm location
    # base_path = '/home/justin/Desktop/Davis_Data_Backup/Wednesday/thick_10cmFoV_mmsteps/'
    # files = ['2020-10-07-1210.h5']

    location = "Davis"  # was Berkeley (Davis, Berkeley, Fix)
    filepaths = [base_path + file for file in files]
    full_run = system_processing(filepaths, place=location, mod_adc_max_bin=140000,  # 100000 dflt
                                 mod_adc_bin_size=150,  # 150 dflt
                                 pmt_adc_max_bin=100000)  # 80000 dflt

    module_hist_bins = np.linspace(0, full_run.runs[0].mod_histogram_bins[-1],
                                   full_run.runs[0].mod_histogram_bins.size - 1)  # mods

    module_hist_bins[1:] = 4.21 * np.log(module_hist_bins[1:]) - 40.8  # TODO (NOTE): Delete for just ADC values

    # === Rough Calibration ===
    mod_calib = np.array([4.8, 5.06, 4.77, 4.73,  # beam on
                          4.69, 5.03, 5.02, 4.82,
                          5.34, 4.78, 5.16, 4.97,
                          4.38, 4.24, 4.85, 4.45])
    calib_beam_factor = 10.5 / 11  # 10.5/11  # This accounts for average gain shift relative to beam off (Th-228 data)
    rel_calib_factor = np.array([1, 1, 1, 1,
                                 0.98, 1, 1, 1.025,
                                 1, 1, 1, 1.03,
                                 0.97, 0.94, 0.955, 1])  # relative to SID 9

    full_run.dyn_mod_gains = mod_calib.mean() / mod_calib * calib_beam_factor * rel_calib_factor
    # === Rough Calibration ===

    # ====== Pixel Mask =====
    m = np.zeros([12, 12])
    # m[4:8, 4:8] = 1
    m[3:9, 3:9] = 1
    # print(m)
    pts = np.argwhere(m)
    pixel_masks = create_mask(type='middle', single_pxls=pts)
    print(pixel_masks)
    # ===== Pixel Masks =====

    # e_filter = [30000, 55000]  # Feb 15, March 16 [20000, 80000], Apr 12 [30000, 55000] i.e. C, SE, and DE
    e_filter = [30000, 180000]  # full spectrum
    # e_filter = [30000, 43000]  # carbon_scatter
    # e_filter = [43000, 50000]  # carbon
    # e_filter = [50000, 53000]  # oxygen_scatter
    # e_filter = [53000, 75000]  # oxygen

    num_tbins = 10

    time_module_histograms = np.zeros([num_tbins, module_hist_bins.size])  # Storage for each time bin
    tbin_labels = np.zeros([num_tbins, 2])

    for id, percent in enumerate(1/num_tbins * np.arange(num_tbins)):
        # if not id == 0:
        #    continue
        (tmin, tmax), time_pts = time_select_indices(percent, percent+0.1)
        full_run.generate_spectra(filter_limits=e_filter, choose_mods=mod, time_modulo=time_pts, masks=pixel_masks)

        fig, axes = full_run.display_spectra_and_image(mod_id=mod,
                                                      image_name='{:.2f} to {:.2f} ns'.format(tmin, tmax))
        print("Total Events: ", full_run.module_histograms.sum())

        time_module_histograms[id] = np.copy(full_run.module_histograms[mod])
        tbin_labels[id] = (tmin, tmax)
        # if id == 0:
        #     plt.show()
        plt.close(fig)

    # print("tbin_labels: ", tbin_labels)

    # fig, axes = full_run.display_spectra_and_image(image_name='Hi Justin')
    # print("Total Events: ", full_run.module_histograms.sum())

    for run in full_run.runs:
        run.h5file.close()

    # Plot Different Time Bins
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colormap = plt.cm.nipy_spectral(np.linspace(0, 1, num_tbins))
    ax.set_prop_cycle('color', colormap)
    for tbin, (t_histogram, t_labels) in enumerate(zip(time_module_histograms, tbin_labels)):
        # image_name = '{:.2f} to {:.2f} ns'.format(tmin, tmax))
        lbl = '{:.2f} to {:.2f} ns'.format(t_labels[0], t_labels[1])
        ax.step(module_hist_bins, t_histogram, where='mid', label=lbl)

    ax.set_yscale('log')
    ax.set_title('Module Spectrum', fontsize=24)

    ll = np.min(np.argmax(time_module_histograms > 0, axis=1))
    ul = time_module_histograms.shape[1] - np.min(np.argmax(np.fliplr(time_module_histograms) > 0, axis=1)) - 1

    ax.set_xlim([np.max([0, 0.9 * module_hist_bins[ll]]), np.min([1.01 * module_hist_bins[ul], 1.01 * module_hist_bins[-1]])])
    ax.legend(loc='upper right', prop={'size': 14})
    # ax.xticks(fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    plt.show()


if __name__ == "__main__":
    # process_projection()
    process_projection_mod(9)  # 9 is our guy

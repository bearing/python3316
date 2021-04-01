import matplotlib.pyplot as plt
import numpy as np
from legacy.single_module_processing import events_recon
from legacy.one_module_processing import events_recon as per


class system_projection(object):  # use with single_module_processing

    def __init__(self, filepaths):
        if type(filepaths) == str:
            files = [filepaths]
        else:
            files = filepaths

        self.runs = []
        for file in files:
            self.runs.append(events_recon(file))

        self.mapped = np.zeros([48, 48])  # total from runs
        self.system_id = np.arange(16)  # when facing front of detectors, upper left to upper right, then down
        self.mod_id = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])  # The order they were
        # plugged in by channel id

    def run_mod_process(self, sid, **kwargs):
        run_proj = np.zeros([12, 12])
        total_energy_spectra = np.zeros(self.runs[0].histogram_bins.size-1)

        for run in self.runs:
            eng, _, crude_bin = run.projection_binned(4 * self.mod_id[sid], **kwargs)
            total_energy_spectra += eng
            run_proj += crude_bin

        return total_energy_spectra, run_proj

    def complete_run_proj(self, **kwargs):  # kwarg c, off, and energy_filter
        for sid in self.system_id:
            mhist, mproj= self.run_mod_process(sid, **kwargs)
            yield sid, mhist, mproj
            # self.mapped[(row * 12): (row + 1) * 12, (col * 12): (col + 1) * 12] = crude_bin

    def display_projections(self, **kwargs):  # save_fname=None
        fig, (ax1, ax2) = plt.subplots(1, 2)
        x_e = 2 * np.log(np.linspace(0, self.runs[0].histogram_bins[-1], self.runs[0].histogram_bins.size-1)) - 17.5
        for sid, mod_hist, mod_proj in self.complete_run_proj(**kwargs):
            print("System ID ", str(sid), "(Module ID ", str(self.mod_id[sid]), ") Processed")
            row = sid // 4
            col = sid % 4
            self.mapped[(row * 12): (row + 1) * 12, (col * 12): (col + 1) * 12] = mod_proj
            ax1.step(x_e, mod_hist, label='mod' + str(self.mod_id[sid]))
        ax1.set_yscale('log')
        ax1.set_xlim([2, 1.01 * np.max(x_e)])
        ax1.legend(loc='best')
        ax2.imshow(self.mapped, cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')

        fig.tight_layout()
        plt.show()

        # if type(save_fname) is str:
        #    np.save(save_fname, self.mapped)


class system_processing(object):

    def __init__(self, filepaths, place="Davis"):
        if type(filepaths) == str:
            files = [filepaths]
        else:
            files = filepaths

        self.runs = []
        for file in files:
            self.runs.append(per(file, place=place))

        self.mapped = np.zeros([48, 48])  # total from runs
        self.system_id = np.arange(16)  # when facing front of detectors, upper left to upper right, then down
        if place == "Davis":
            self.mod_id = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])  # The order they were
        # plugged in by channel id
            print("Hi Davis")
        else:
            self.mod_id = np.arange(16)

    def run_mod_process(self, rid, sid, **kwargs):  # system id, run id
        run_proj = np.zeros([12, 12])
        total_energy_spectra = np.zeros(self.runs[0].histogram_bins.size-1)

        for run in self.runs:
            eng, _, crude_bin = run.projection_binned(rid, sid, **kwargs)
            total_energy_spectra += eng
            run_proj += crude_bin

        return total_energy_spectra, run_proj

    def complete_run_proj(self, **kwargs):  # kwarg -> energy_filter
        for sid in self.system_id:
            mhist, mproj= self.run_mod_process(4 * self.mod_id[sid], sid, **kwargs)
            yield sid, mhist, mproj
            # self.mapped[(row * 12): (row + 1) * 12, (col * 12): (col + 1) * 12] = crude_bin

    def display_projections(self, **kwargs):  # save_fname=None
        fig, (ax1, ax2) = plt.subplots(1, 2)
        x_e = 2 * np.log(np.linspace(0, self.runs[0].histogram_bins[-1], self.runs[0].histogram_bins.size-1)) - 17.5
        for sid, mod_hist, mod_proj in self.complete_run_proj(**kwargs):
            print("System ID ", str(sid), "(Module ID ", str(self.mod_id[sid]), ") Processed")
            row = sid // 4
            col = sid % 4
            self.mapped[(row * 12): (row + 1) * 12, (col * 12): (col + 1) * 12] = mod_proj
            ax1.step(x_e, mod_hist, label='mod' + str(self.mod_id[sid]))
        ax1.set_yscale('log')
        ax1.set_xlim([2, 1.01 * np.max(x_e)])
        ax1.legend(loc='best')
        # im = ax2.imshow(self.mapped, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal')
        # original
        im = ax2.imshow(self.mapped.T, cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')

        fig.colorbar(im, fraction= 0.046, pad=0.04, ax=ax2)
        fig.tight_layout()
        plt.show()


def main():
    base_path = '/home/proton/repos/python3316/Data/'
    files = ['2020-10-08-1438.h5', '2020-10-08-1443.h5', '2020-10-08-1447.h5', '2020-10-08-1451.h5',
             '2020-10-08-1456.h5', '2020-10-08-1500.h5', '2020-10-08-1503.h5', '2020-10-08-1507.h5',
             '2020-10-08-1511.h5', '2020-10-08-1515.h5', '2020-10-08-1519.h5', '2020-10-08-1522.h5',
             '2020-10-08-1526.h5', '2020-10-08-1530.h5', '2020-10-08-1534.h5', '2020-10-08-1537.h5']
    # files2 = ['2020-10-31-1704.h5']
    filepaths = [base_path + file for file in files]
    full_run = system_projection(filepaths)
    full_run.display_projections(energy_filter=[2.5])
    for run in full_run.runs:
        run.h5file.close()
    np.save('full_run_bg', full_run.mapped)


def main_th_measurement():
    base_path = '/home/proton/repos/python3316/Data/'
    files = ['2020-10-31-1704.h5']
    location = "Berkeley"
    filepaths = [base_path + file for file in files]
    full_run = system_processing(filepaths, place=location)
    full_run.display_projections(energy_filter=[2.0])
    for run in full_run.runs:
        run.h5file.close()
    # np.save('th_uncalib_Oct31_flood', full_run.mapped)  # 2 MeV or greater for 18 hr Th measurement


def main_nov3():
    base_path = '/home/proton/repos/python3316/Data/'
    # files = ['2020-10-08-1438.h5',  '2020-10-08-1443.h5', '2020-10-08-1447.h5', '2020-10-08-1451.h5',
    #         '2020-10-08-1456.h5']  # 20 mins at Davis, pos 6 far
    files = ['2020-10-08-1542.h5', '2020-10-08-1546.h5', '2020-10-08-1550.h5', '2020-10-08-1554.h5',
             '2020-10-08-1558.h5']  # pos0
    # files2 = ['2020-10-31-1704.h5']
    filepaths = [base_path + file for file in files]
    full_run = system_processing(filepaths)
    full_run.display_projections(energy_filter=[1, 7])
    for run in full_run.runs:
        run.h5file.close()
    # np.save('full_run_pos0A_Nov3', full_run.mapped)
    # pos6A ->  calib['swapped_pmts'] = np.array([[0, 3], [1, 2]])  # probably correct
    # pos6B ->  calib['swapped_pmts'] = np.array([[1, 2], [0, 3]])


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

    filepaths = [base_path + file for file in files12]
    full_run = system_processing(filepaths)
    full_run.display_projections(energy_filter=[3.5, 7])
    for run in full_run.runs:
        run.h5file.close()
    # np.save('step_run_8t9cm_Nov3', full_run.mapped)  # hit start


def main_view_processed_proj():
    base_path = '/home/proton/repos/python3316/processing/step_run_'
    end_path = 'cm_Nov3.npy'
    beg_step = np.arange(1, 10)
    # end_step = beg_step + 1
    # end_step = 0
    for step in beg_step:
        specific_path = str(step) + 't' + str(step + 1)
        loaded_map = np.load(base_path + specific_path + end_path)
        plt.imshow(loaded_map.T, cmap='jet', origin='upper', interpolation='nearest', aspect='equal')
        plt.colorbar()
        plt.title('Position ' + str(step) + '-' + str(step + 1) + ' cm')
        plt.show()
        plt.clf()

    # specific_path = str(beg_step) + 't' + str(end_step)
    # loaded_map = np.load(base_path + specific_path + end_path)
    # Need to flip? Do it here.
    # fig = plt.imshow(loaded_map.T, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal')
    # for stp in beg_step:
    #     pass

    # plt.title()
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    # main()
    main_th_measurement()
    # main_nov3()
    # main_small_steps()
    # main_view_processed_proj()

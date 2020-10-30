import tables
import io
import matplotlib.pyplot as plt
import numpy as np
from processing.single_module_processing import events_recon, load_signals


class system_projection(object):

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
        x_e = 2 * np.log(np.linspace(0, self.runs[0].histogram_bins[-1], self.runs[0].histogram_bins.size-1)) - 17
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


def main():
    base_path = '/home/proton/repos/python3316/Data/'
    files = ['2020-10-08-1438.h5', '2020-10-08-1443.h5', '2020-10-08-1447.h5', '2020-10-08-1451.h5',
             '2020-10-08-1456.h5', '2020-10-08-1500.h5', '2020-10-08-1503.h5', '2020-10-08-1507.h5',
             '2020-10-08-1511.h5', '2020-10-08-1515.h5', '2020-10-08-1519.h5', '2020-10-08-1522.h5',
             '2020-10-08-1526.h5', '2020-10-08-1530.h5', '2020-10-08-1534.h5', '2020-10-08-1537.h5']
    filepaths = [base_path + file for file in files]
    full_run = system_projection(filepaths)
    full_run.display_projections(energy_filter=[2.5])
    for run in full_run.runs:
        run.h5file.close()
    np.save('full_run_6pos_far', full_run.mapped)


if __name__ == "__main__":
    main()
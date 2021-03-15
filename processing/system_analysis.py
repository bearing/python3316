import tables
import io
import matplotlib.pyplot as plt
import numpy as np
from processing.correlated_analysis import events_recon as per
# from processing.calibration_values import load_calibration


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
        else:
            self.mod_id = np.arange(16)

    def run_mod_process(self, rid, sid, **kwargs):  # system id, run id
        run_proj = np.zeros([12, 12])
        total_energy_spectra = np.zeros(self.runs[0].histogram_bins.size-1)

        for run in self.runs:
            eng, _, crude_bin = run.projection_binned(rid, sid, **kwargs)
            if eng is not None:
                total_energy_spectra += eng
                run_proj += crude_bin

        return total_energy_spectra, run_proj

    def complete_run_proj(self, **kwargs):  # kwarg -> energy_filter
        for sid in self.system_id:
            mhist, mproj= self.run_mod_process(self.mod_id[sid], sid, **kwargs)
            yield sid, mhist, mproj
            # self.mapped[(row * 12): (row + 1) * 12, (col * 12): (col + 1) * 12] = crude_bin

    def display_projections(self, index=None, **kwargs):  # save_fname=None
        # print("kargs time_filter? ", kwargs['time_filter'])
        if 'time_filter' in kwargs:
            print("It worked!")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
        x_e = 2 * np.log(np.linspace(0, self.runs[0].histogram_bins[-1], self.runs[0].histogram_bins.size-1)) - 17.5
        for sid, mod_hist, mod_proj in self.complete_run_proj(**kwargs):
            # if sid == 6:
            #    continue
            print("System ID ", str(sid), "(Module ID ", str(self.mod_id[sid]), ") Processed")
            row = sid // 4
            col = sid % 4
            self.mapped[(row * 12): (row + 1) * 12, (col * 12): (col + 1) * 12] = mod_proj
            ax1.step(x_e, mod_hist, label='mod' + str(self.mod_id[sid]))
        ax1.set_yscale('log')
        ax1.set_xlim([2, 1.01 * np.max(x_e)])
        ax1.legend(loc='best')
        ax1.set_xlabel("Energy (Mev)")

        ax2.imshow(self.mapped, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal')
        ax2.set_xticks([])
        ax2.set_yticks([])
        total_counts = np.sum(self.mapped).astype(int)

        if 'energy_filter' not in kwargs:
            en_limits = [2, 'above']
        else:
            en_limits = np.array(kwargs['energy_filter'])

        if 'time_filter' in kwargs:
            time_range = 4 * np.sort(kwargs['time_filter'])
        else:
            time_range = np.array([-48, 48])

        ax2.set_title("Projection {f} to {end} ns from Plastic Trigger \n"
                      "Total Counts: {tc} ({e1} to {e2} MeV)".format(f=time_range[0],
                                                                     end=time_range[1],
                                                                     tc=total_counts,
                                                                     e1=en_limits[0],
                                                                     e2=en_limits[1]))

        fig.tight_layout()
        if index:
            save_fname = 'batch/image' + str(index) + '.png'
            plt.savefig(save_fname, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def main():
    base_path = '/home/proton/repos/python3316/processing/processedW-48+48_10-08-'
    files = ['1522.h5', '1526.h5', '1530.h5', '1534.h5', '1537.h5']  # Position 6, far

    filepaths = [base_path + file for file in files]
    full_run = system_processing(filepaths)
    # full_run.display_projections(energy_filter=[3.5, 7])
    # time_bins = np.array([0, 0.5])
    time_edges = np.arange(-24, 25)/2
    for edge in np.arange(time_edges.size - 1):
        time_bins = time_edges[edge:edge+2]
        full_run.display_projections(index=edge,energy_filter=[3.5, 7], time_filter=time_bins)

    # full_run.display_projections(energy_filter=[3.5, 7], time_filter=time_bins)
    # savefig('foo.png', bbox_inches='tight')
    for run in full_run.runs:
        run.h5file.close()
    # np.save('full_run_pos0A_Nov3', full_run.mapped)
    # pos6A ->  calib['swapped_pmts'] = np.array([[0, 3], [1, 2]])  # probably correct
    # pos6B ->  calib['swapped_pmts'] = np.array([[1, 2], [0, 3]])


if __name__ == "__main__":
    main()
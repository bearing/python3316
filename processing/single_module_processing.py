import tables
import io
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

# '/Users/justinellin/Desktop/Davis 2020/Tuesday/2020-10-06-1503.h5'
# -> beam axis is +x-hat, away from earth is +y-hat


class events_recon(object):
    _crystal_coordinates = 'peak_coords_mean.txt'
    _energy_coeff_a = 2  # This is how you would fit for each module
    _energy_coeff_b = 17

    def __init__(self, filepath):
        self.h5file = load_signals(filepath)
        self.crd = load_coordinates(self._crystal_coordinates)
        self.pxl_mapper = pixel_mapper(self.crd)
        self.histogram_bins = np.arange(0, 180000, 1000)
        # self.histogram_bins = np.arange(0, 30000, 1000)  # For background
        self.gamma_events = 0
        self.filtered_gamma_events = 0
        self.crude_crystal_cutsX = np.array([0, 14, 21, 27, 33, 40, 50, 58, 65, 72, 79, 86, 100])
        self.crude_crystal_cutsY = np.array([0, 14, 21, 27, 33, 40, 50, 58, 65, 72, 79, 86, 100])
        self.orig_hist_size = np.array([100, 100])
        self.energy_coeff_a = 2  # This is how you would fit for each module
        self.energy_coeff_b = 17

    def projection(self, start_index, c=np.ones(4), off=np.zeros(4)):
        module = np.arange(4)
        tables = [0 for _ in np.arange(4)]
        energy_process = [0 for _ in np.arange(4)]

        for integer in module:
            folder = '/det' + str(int(start_index + integer))
            tables[integer] = self.h5file.get_node('/', folder).EventData
        evts = tables[0].nrows
        energy_array = np.zeros(evts, dtype=np.float)
        for channel in module:
            energy_process[channel] = energy_array.copy()
        if evts < 1:
            return None, None
        self.gamma_events += evts

        # ch0 -> LL, ch1 -> UL, ch2-> LR, ch3 -> UR with x-> beam-direction, y -> skyward
        # ch0 = L4 = LL, ch1 = L3 = UL, ch2 = L2 = LR, ch3 = L1 = UR
        if (start_index//4) != 9 and (start_index//4) != 2:
            energy_process[0] = c[0] * tables[0].col('gate2') - 3.0 * tables[0].col('gate1') - off[0]
            energy_process[1] = c[1] * tables[1].col('gate2') - 3.0 * tables[1].col('gate1') - off[1]
            energy_process[2] = c[2] * tables[2].col('gate2') - 3.0 * tables[2].col('gate1') - off[2]
            energy_process[3] = c[3] * tables[3].col('gate2') - 3.0 * tables[3].col('gate1') - off[3]
        else:
            if (start_index // 4) == 9:
                # module_ind 9 swapped channel 0 and 1 or 2 and 3. TODO: FIND WHICH
                energy_process[0] = c[0] * tables[0].col('gate2') - 3.0 * tables[0].col('gate1') - off[0]
                energy_process[1] = c[1] * tables[1].col('gate2') - 3.0 * tables[1].col('gate1') - off[1]
                energy_process[2] = c[2] * tables[3].col('gate2') - 3.0 * tables[3].col('gate1') - off[2]
                energy_process[3] = c[3] * tables[2].col('gate2') - 3.0 * tables[2].col('gate1') - off[3]
            else:
                E1 = c[0] * tables[0].col('gate2') - 3.0 * tables[0].col('gate1') - off[0]
                E2 = c[1] * tables[1].col('gate2') - 3.0 * tables[1].col('gate1') - off[1]
                E3 = c[2] * tables[2].col('gate2') - 3.0 * tables[2].col('gate1') - off[2]
                E4 = 0.63 * c[3] * tables[3].col('gate2') - 3.0 * tables[3].col('gate1') - off[3] + 2000
                # E4 = 1 * c[3] * tables[3].col('gate2') - 3.0 * tables[3].col('gate1') - off[3]
                # E4 multiplier = 0.63, offset = 2000

                energy_process[0] = E1
                energy_process[1] = E2
                energy_process[2] = E3
                energy_process[3] = E4

        sum_en = energy_process[0] + energy_process[1] + energy_process[2] + energy_process[3]

        Ex = ((energy_process[2] - energy_process[0]) + (energy_process[3] - energy_process[1])) / (1.0 * sum_en)
        Ey = ((energy_process[1] - energy_process[0]) + (energy_process[3] - energy_process[2])) / (1.0 * sum_en)

        if start_index//4 == 2:
            binX = np.linspace(-1.4, 0.9, 201)
            binY = np.linspace(-1.4, 0.9, 201)
        else:
            binX = np.linspace(-1.0, 1.0, 101)
            binY = np.linspace(-1.0, 1.0, 101)

        original_hist = np.histogram2d(Ex, Ey, bins=[binX, binY])[0]

        self.orig_hist_size = np.array([binX.size, binY.size])

        energy_hist = np.histogram(sum_en, bins=self.histogram_bins)[0]
        E1 = np.histogram(energy_process[0], bins=self.histogram_bins)[0]
        E2 = np.histogram(energy_process[1], bins=self.histogram_bins)[0]
        E3 = np.histogram(energy_process[2], bins=self.histogram_bins)[0]
        E4 = np.histogram(energy_process[3], bins=self.histogram_bins)[0]

        return [energy_hist, E1, E2, E3, E4], original_hist

    def projection_binned(self, start_index, c=np.ones(4), off=np.zeros(4), energy_filter=[]):
        module = np.arange(4)
        tables = [0 for _ in np.arange(4)]
        energy_process = [0 for _ in np.arange(4)]

        for integer in module:
            folder = '/det' + str(int(start_index + integer))
            tables[integer] = self.h5file.get_node('/', folder).EventData
        evts = tables[0].nrows
        energy_array = np.zeros(evts, dtype=np.float)
        for channel in module:
            energy_process[channel] = energy_array.copy()
        if evts < 1:
            return None, None
        self.gamma_events += evts

        # ch0 -> LL, ch1 -> UL, ch2-> LR, ch3 -> UR with x-> beam-direction, y -> skyward
        # ch0 = L4 = LL, ch1 = L3 = UL, ch2 = L2 = LR, ch3 = L1 = UR
        if (start_index//4) != 9 and (start_index//4) != 2:
            energy_process[0] = c[0] * tables[0].col('gate2') - 3.0 * tables[0].col('gate1') - off[0]
            energy_process[1] = c[1] * tables[1].col('gate2') - 3.0 * tables[1].col('gate1') - off[1]
            energy_process[2] = c[2] * tables[2].col('gate2') - 3.0 * tables[2].col('gate1') - off[2]
            energy_process[3] = c[3] * tables[3].col('gate2') - 3.0 * tables[3].col('gate1') - off[3]
        else:
            if (start_index // 4) == 9:
                # module_ind 9 swapped channel 0 and 1 or 2 and 3. TODO: FIND WHICH
                energy_process[0] = c[0] * tables[0].col('gate2') - 3.0 * tables[0].col('gate1') - off[0]
                energy_process[1] = c[1] * tables[1].col('gate2') - 3.0 * tables[1].col('gate1') - off[1]
                energy_process[2] = c[2] * tables[3].col('gate2') - 3.0 * tables[3].col('gate1') - off[2]
                energy_process[3] = c[3] * tables[2].col('gate2') - 3.0 * tables[2].col('gate1') - off[3]
            else:
                # print("Hi")
                E1 = c[0] * tables[0].col('gate2') - 3.0 * tables[0].col('gate1') - off[0]
                E2 = c[1] * tables[1].col('gate2') - 3.0 * tables[1].col('gate1') - off[1]
                E3 = c[2] * tables[2].col('gate2') - 3.0 * tables[2].col('gate1') - off[2]
                E4 = 0.63 * c[3] * tables[3].col('gate2') - 3.0 * tables[3].col('gate1') - off[3] + 2000
                # E4 = 1 * c[3] * tables[3].col('gate2') - 3.0 * tables[3].col('gate1') - off[3]
                # E4 multiplier = 0.63, offset = 2000

                energy_process[0] = E1
                energy_process[1] = E2
                energy_process[2] = E3
                energy_process[3] = E4

        orig_sum_en = energy_process[0] + energy_process[1] + energy_process[2] + energy_process[3]

        origEx = ((energy_process[2] - energy_process[0]) + (energy_process[3] - energy_process[1])) /\
                 (1.0 * orig_sum_en)
        origEy = ((energy_process[1] - energy_process[0]) + (energy_process[3] - energy_process[2])) / \
                 (1.0 * orig_sum_en)

        valid_ind = (orig_sum_en > 0)

        valid_Ex = origEx[valid_ind]
        valid_Ey = origEy[valid_ind]
        valid_E_sum = orig_sum_en[valid_ind]

        sum_mev = 2 * np.log(1.0 * valid_E_sum) - 17

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

        if start_index//4 == 2:
            Ex += 0.05
            Ey += 0.05

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

        if (start_index//4) not in (2, 5, 14):  # i.e. defaults
            self.crude_crystal_cutsX = np.array([0, 14, 21, 27, 33, 40, 50, 58, 65, 72, 79, 86, 100])
            self.crude_crystal_cutsY = np.array([0, 14, 21, 27, 33, 40, 50, 58, 65, 72, 79, 86, 100])

        if (start_index//4) == 2:  # to do
            self.crude_crystal_cutsX = np.array([0, 12, 20, 27.5, 32, 42, 51, 60, 66.5, 74.5, 82, 89, 100])
            self.crude_crystal_cutsY = np.array([0, 12.5, 20, 27, 32.25, 41, 50.5, 60, 67, 73, 81, 87.5, 100])

        if (start_index//4) == 5:
            self.crude_crystal_cutsX = np.array([0, 14, 19.5, 25.5, 31, 39, 48, 57, 65, 70, 77, 83, 100])
            self.crude_crystal_cutsY = np.array([0, 13, 21, 27, 33, 40, 50, 58, 65, 72, 79, 83, 100])

        if (start_index // 4) == 14:
            self.crude_crystal_cutsX = np.array([0, 14, 19.5, 25.5, 31, 39, 48, 57, 65, 70, 77, 82, 100])
            self.crude_crystal_cutsY = np.array([0, 15, 21, 27, 33, 40, 50, 58, 65, 72, 79, 83, 100])

        crude_binning = np.histogram2d(Ex_scaled, Ey_scaled,
                                       bins=[self.crude_crystal_cutsX, self.crude_crystal_cutsY])[0]

        return energy_hist, original_hist, crude_binning


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


def load_coordinates(calib_file):
    f = open(calib_file, 'r')
    coordinates = eval(f.read().replace(' ', ','))
    f.close()
    return np.array(coordinates)


def pixel_mapper(crds):
    roots = crds[:, :2]  # need to flip, John X is along skyward axis, y along beam. Opposite to mine
    roots[:, [0, 1]] = roots[:, [1, 0]]
    roots[:, 1] *= -1
    roots[:, 1] += 100
    # print('Data points for Tree:, ', spatial.cKDTree(roots).n)
    return spatial.cKDTree(roots)


def main():

    # file = '/home/proton/repos/python3316/Data/2020-10-08-1438.h5'
    # file = '/home/proton/repos/python3316/Data/2020-10-08-1132.h5'  # Position 12
    # file = '/home/proton/repos/python3316/Data/2020-10-08-1503.h5'  # Position 6, far
    # file = '/home/proton/repos/python3316/Data/2020-10-08-1554.h5'  # Position 0, far
    # Overnight: 1744
    # file = '/home/proton/repos/python3316/Data/2020-10-07-1744.h5'

    file = '/home/proton/repos/python3316/Data/2020-10-31-1704.h5'
    tst = events_recon(file)

    # projection = np.zeros([100, 100])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    mod_id = 4

    eng, proj = tst.projection(4 * mod_id)

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

    # file = '/home/proton/repos/python3316/Data/2020-10-08-1438.h5'
    # file = '/home/proton/repos/python3316/Data/2020-10-08-1132.h5'  # Position 12
    # file = '/home/proton/repos/python3316/Data/2020-10-08-1503.h5'  # Position 6, far
    # file = '/home/proton/repos/python3316/Data/2020-10-08-1554.h5'  # Position 0, far
    # Overnight: 1744
    # file = '/home/proton/repos/python3316/Data/2020-10-07-1744.h5'
    # background
    file = '/home/proton/repos/python3316/Data/2020-10-31-1704.h5'
    tst = events_recon(file)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # mod_id = 2  # Gain Problems
    mod_id = 15

    eng, proj, binned = tst.projection_binned(4 * mod_id, energy_filter=[])

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
    main()  # unbinned
    # main_binned()
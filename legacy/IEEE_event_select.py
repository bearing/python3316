import tables
import io
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

# '/Users/justinellin/Desktop/Davis 2020/Tuesday/2020-10-06-1503.h5'
# -> beam axis is +x-hat, away from earth is +y-hat


class events_recon(object):
    _crystal_coordinates = 'peak_coords_mean.txt'

    def __init__(self, filepath):
        self.h5file = load_signals(filepath)
        self.crd = load_coordinates(self._crystal_coordinates)
        self.pxl_mapper = pixel_mapper(self.crd)
        self.histogram_bins = np.arange(0, 100000, 1000)
        self.gamma_events = 0
        self.crude_crystal_cuts = np.array([0, 14, 21, 27, 33, 40, 50, 58, 65, 72, 79, 86, 100])

    def projection(self, start_index):
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
        energy_process[0] = tables[0].col('gate2') - 3.0 * tables[0].col('gate1')
        energy_process[1] = tables[1].col('gate2') - 3.0 * tables[1].col('gate1')
        energy_process[2] = tables[2].col('gate2') - 3.0 * tables[2].col('gate1')
        energy_process[3] = tables[3].col('gate2') - 3.0 * tables[3].col('gate1')

        sum_en = energy_process[0] + energy_process[1] + energy_process[2] + energy_process[3]

        Ex = ((energy_process[2] - energy_process[0]) + (energy_process[3] - energy_process[1])) / (1.0 * sum_en)
        Ey = ((energy_process[1] - energy_process[0]) + (energy_process[3] - energy_process[2])) / (1.0 * sum_en)

        binX = np.linspace(-1.0, 1.0, 101)
        binY = np.linspace(-1.0, 1.0, 101)

        original_hist = np.histogram2d(Ex, Ey, bins=[binX, binY])[0]

        energy_hist = np.histogram(sum_en, bins=self.histogram_bins)[0]

        Ex_scaled = (Ex + 1)/0.02
        Ey_scaled = (Ey + 1)/0.02

        Ex_scaled[Ex_scaled > 100] = 99
        Ey_scaled[Ey_scaled > 100] = 99

        crude_binning = np.histogram2d(Ex_scaled, Ey_scaled, bins=[self.crude_crystal_cuts, self.crude_crystal_cuts])[0]

        if (start_index//4) not in (5, 14):
            crude_binning = np.histogram2d(Ex_scaled, Ey_scaled,
                                           bins=[self.crude_crystal_cuts, self.crude_crystal_cuts])[0]
        else:
            if (start_index//4) == 5:
                mod5binsx = np.array([0, 14, 19.5, 25.5, 31, 39, 48, 57, 65, 70, 77, 83, 100])
                mod5binsy = np.array([0, 13, 21, 27, 33, 40, 50, 58, 65, 72, 79, 83, 100])
                crude_binning = np.histogram2d(Ex_scaled, Ey_scaled,
                                               bins=[mod5binsx,  mod5binsy])[0]
            else:
                mod14binsx = np.array([0, 14, 19.5, 25.5, 31, 39, 48, 57, 65, 70, 77, 82, 100])
                mod14binsy = np.array([0, 15, 21, 27, 33, 40, 50, 58, 65, 72, 79, 83, 100])
                crude_binning = np.histogram2d(Ex_scaled, Ey_scaled,
                                               bins=[mod14binsx, mod14binsy])[0]
        # self.crude_crystal_cuts = np.array([0, 14, 21, 27, 33, 40, 50, 58, 65, 72, 79, 86, 100])
        # self.pxl_mapper.query()

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

    file = '/home/proton/repos/python3316/Data/2020-10-08-1148.h5'
    tst = events_recon(file)

    # TODO: This should be temporary
    # projection = np.zeros([48, 48])
    projection = np.zeros([100, 100])

    # module_ind = np.arange(1)
    module_ind = np.array([6])
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set(adjustable='box-forced', aspect='equal')
    ax2.set(adjustable='box-forced', aspect='equal')
    ax3.set(adjustable='box-forced', aspect='equal')
    for idx in module_ind:
        print("Current Index: ", idx)

        eng, proj = tst.get_histograms(4 * idx)  # TODO: Remove when fixed

        if eng is not None and proj is not None:
            x = np.linspace(0, 100000, eng.size)
            projection = proj
            ax1.step(x, eng, label='mod'+str(idx))
    tst.h5file.close()
    print("Total Gamma Events:", tst.gamma_events)
    ax1.set_yscale('log')
    ax1.legend(loc='best')
    im = ax2.imshow(projection, cmap='viridis', interpolation='nearest')

    crystal_pts = load_coordinates('peak_coords_mean.txt')
    ax3.scatter(crystal_pts[:12, 0], 100-crystal_pts[:12, 1])
    ax3.set_ylim((0, 100))
    ax3.set_xlim((0, 100))

    # fig.colorbar(im, ax=ax2)
    plt.show()


def main2():

    # file = '/home/proton/repos/python3316/Data/2020-10-08-1438.h5'
    # file = '/home/proton/repos/python3316/Data/2020-10-08-1132.h5'  # Position 12
    file = '/home/proton/repos/python3316/Data/2020-10-08-1503.h5'  # Position 6, far
    # file = '/home/proton/repos/python3316/Data/2020-10-08-1554.h5'  # Position 0, far
    # Overnight: 1744
    # file = '/home/proton/repos/python3316/Data/2020-10-07-1744.h5'
    tst = events_recon(file)

    gain_cor = np.load('correction.npy')
    # print("Gain_cor size:", gain_cor.shape)

    # projection = np.zeros([100, 100])
    summed_projection = np.zeros([100, 100])
    mapped = np.zeros([48, 48])

    # module_ind = np.arange(16)[::1]
    # module_ind = np.array([12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3])
    module_ind = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    # List of modules starting from upper left (facing collimator) across to upper right, then down
    # module_ind = np.array([1])
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, (ax2, ax3) = plt.subplots(1, 2)
    # ax1.set(adjustable='box-forced', aspect='equal')
    ax2.set(adjustable='box-forced', aspect='equal')
    ax3.set(adjustable='box-forced', aspect='equal')

    row = 0
    col = 0
    for idx, mod_id in enumerate(module_ind):
        print("Current Index: ", mod_id)
        row = idx // 4
        col = idx % 4
        eng, proj, crude_bin = tst.projection(4 * mod_id)  # TODO: Remove when fixed

        if mod_id == 9 or mod_id == 2:  # was idx ==14, mod 2 is also ignored
            continue
        if eng is not None and proj is not None:
            x = np.linspace(0, 100000, eng.size)
            if mod_id == 14:  # mod_id 5
                summed_projection += proj
                # ax1.step(x, eng, label='mod' + str(mod_id))
            # summed_projection += proj
            mapped[(row * 12): (row+1) * 12, (col * 12): (col+1) * 12] = crude_bin
            # ax1.step(x, eng, label='mod' + str(mod_id))
    tst.h5file.close()
    print("Total Gamma Events:", tst.gamma_events)
    # ax1.set_yscale('log')
    # ax1.legend(loc='best')
    im = ax2.imshow(summed_projection, cmap='viridis', origin='lower', interpolation='nearest')
    im2 = ax3.imshow(mapped, cmap='jet', interpolation='none')

    # print("Mean of mapped:", np.mean(mapped[mapped>0]))
    # print("Median of mapped:", np.median(mapped[mapped>0]))
    # median = np.median(mapped[mapped>0])
    # mapped[mapped < (median/10)] = (median/10)
    # correction = (median/mapped)
    # correction_fname = 'correction.npy'  # made with 1744 overnight
    # np.save(correction_fname, correction)

    # ax3.set_ylim((0, 100))
    # ax3.set_xlim((0, 100))
    # ax1.hist(mapped.ravel(), bins=500, range=(0, 5000))

    # fig.colorbar(im, ax=ax2)
    plt.show()


if __name__ == "__main__":
    # main()
    main2()
    # main3()

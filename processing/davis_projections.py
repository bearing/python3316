import tables
import io
import imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
# import matplotlib as mpl
import matplotlib.colors as mcolors


class Projections(object):
    _crystal_coordinates = 'peak_coords_mean.txt'

    def __init__(self, filepath):
        self.hf5 = [load_data(file) for file in filepath]
        self.f_idx = np.arange(len(self.hf5))
        self.crds = load_coordinates(self._crystal_coordinates)
        self.pxl_map = pixel_mapper(self.crds)
        # print("First data file nrows:", self.hf5[0].root.event_data.nrows)
        self.evt_data = [self.hf5[idx].root.event_data for idx in self.f_idx]

        # self.lso_data_fields =1 ['bid', 'mod_id', 'ts', 'rel_ts', 'E1', 'E2', 'E3', 'E4']
        # self.lso_data_types = [np.uint32, np.uint8, np.uint64, np.float32, np.uint32, np.uint32, np.uint32, np.uint32]

    def com_pts(self, t_filt=None):
        if t_filt is None:
            t_filt = [-12, 12]

        ts_mask = [(self.evt_data[idx].col('rel_ts')[:] < t_filt[0]) &
                   (self.evt_data[idx].col('rel_ts')[:] < t_filt[1])
                   for idx in self.f_idx]

        E1 = np.array([self.evt_data[idx].col('E1')[ts_mask[idx]] for idx in self.f_idx])
        E2 = np.array([self.evt_data[idx].col('E2')[ts_mask[idx]] for idx in self.f_idx])
        E3 = np.array([self.evt_data[idx].col('E3')[ts_mask[idx]] for idx in self.f_idx])
        E4 = np.array([self.evt_data[idx].col('E4')[ts_mask[idx]] for idx in self.f_idx])

        sum = E1 + E2 + E3 + E4

        # Ex = ((E1 - E3) + (E2 - E4)).astype('float64') / sum.astype('float64')
        # Ey = ((E1 - E2) + (E3 - E4)).astype('float64') / sum.astype('float64')

        return sum

    def scatter_pts(self):

        rel_ts = np.concatenate([self.evt_data[idx].col('rel_ts')[:] for idx in self.f_idx])

        E1 = np.concatenate([self.evt_data[idx].col('E1')[:] for idx in self.f_idx])
        E2 = np.concatenate([self.evt_data[idx].col('E2')[:] for idx in self.f_idx])
        E3 = np.concatenate([self.evt_data[idx].col('E3')[:] for idx in self.f_idx])
        E4 = np.concatenate([self.evt_data[idx].col('E4')[:] for idx in self.f_idx])

        # print("E1 energies:", E[:5])
        # print("E1 energies:", 2 * np.log(1.0 * E1[:5]) - 17.5)
        summed = 1.0 * (E1 + E2 + E3 + E4)
        # print("Sum.shape", summed.shape)

        keep_ind = (summed > 0)
        rel_ts_filter = rel_ts[keep_ind]
        sum_filter = summed[keep_ind]
        # sum_mev = sum_filter
        sum_mev = 2 * np.log(1.0* sum_filter) - 17.5

        # equation = (2, 17.5)

        # Ex = ((E1 - E3) + (E2 - E4)).astype('float64') / sum.astype('float64')
        # Ey = ((E1 - E2) + (E3 - E4)).astype('float64') / sum.astype('float64')
        return rel_ts_filter, sum_mev


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
    base_path = '/home/proton/repos/python3316/processing/'
    # files = ['2020-10-08-1205.h5', '2020-10-08-1522.h5', '2020-10-08-1526.h5', '2020-10-08-1530.h5',
    #         '2020-10-08-1534.h5', '2020-10-08-1537.h5']  # reprocessed
    file_times = [1522, 1526, 1530, 1534, 1537]
    filepaths = [base_path + 'processedW-48+48_10-08-' + str(file_times[f_idx]) + '.h5'
                 for f_idx in np.arange(len(file_times))]

    proj = Projections(filepaths)
    rel_ts, energy = proj.scatter_pts()
    # print('Rel_ts.shape', rel_ts.shape)
    # hist_bins = np.linspace(0, 8, 1000)

    # energy_hist = np.histogram(energy, bins=2000)[0]
    # x = np.linspace(0, np.max(energy), 2000)
    # plt.step(x, energy_hist)
    # plt.yscale('log')

    # plt.scatter(rel_ts* 4, energy, marker=',')
    # rngY = [np.max(energy)/2, np.max(energy)]
    # rngX = [-12, 12]

    # plt.hist2d(rel_ts, energy, bins=[24, 1000], cmap='gist_gray')
    # energy_select = (energy>4) & (energy<7)
    energy_select = (energy>3) & (energy < 7)
    plt.hist2d(rel_ts[energy_select] * 4, energy[energy_select], bins=[96, 20], norm=mcolors.LogNorm(), cmap='gist_gray')
    plt.colorbar()
    plt.xlabel('relative (ns)')
    plt.ylabel('MeV')
    plt.show()


if __name__ == "__main__":
    main()



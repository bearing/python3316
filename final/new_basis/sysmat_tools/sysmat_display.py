import numpy as np
import tables
import os
from datetime import datetime
import time
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def load_sysmat(sysmat_fname):
    if tables.is_hdf5_file(sysmat_fname):
        sysmat_file_obj = load_h5file(sysmat_fname)
        return sysmat_file_obj, sysmat_file_obj.root.sysmat[:].T
    return np.load(sysmat_fname).T


def load_h5file(filepath):  # h5file.root.sysmat[:]
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def makedirs(path):
    """ Create directories for `path` (like 'mkdir -p'). """
    if not path:
        return
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


class sysmat_display(object):

    def __init__(self, fname, dims=(201, 61), center=(0, -10), pxl_sze=1):
        """fname is sysmat file name, dims is (nx, ny), center is center of the FoV, pxl_sze is size of pixels"""
        self.x_dim, self.y_dim = np.array(dims).astype('int')  # just making sure
        self.center = center  # image center
        self.measurements = 48 * 48  # detector pixels
        self.pixels = self.x_dim * self.y_dim
        self.pxl_size = pxl_sze
        self.x_values = ((np.arange(self.x_dim) - (self.x_dim//2)) * pxl_sze) + center[0]
        self.y_values = ((np.arange(self.y_dim)[::-1] - (self.y_dim//2)) * pxl_sze) + center[1]
        self.proj_stack = np.zeros([1, 48, 48])

        self.sysmat_fname = os.path.splitext(fname)[0]  # Remove extension aka .h5
        # Sysmat load and cleanup min value
        self.sysmat_file_obj, self.sysmat = load_sysmat(fname)
        # self.sysmat = load_sysmat(fname).T  # rows = image points, cols = detector measurements
        self.sysmat = self.sysmat.T
        self.sysmat[self.sysmat == 0] = np.min(self.sysmat[self.sysmat != 0])

        self._layer_x_vals = np.array([0])
        self.width = 1

    def generate_projection_stack(self, path_length, start=(0, -10), width=10):
        """path_length is number of steps, start is the (x, y) of where to start, width is width of band."""
        start_row = np.argwhere(self.y_values == np.int(start[1]))
        start_col = np.argwhere(self.x_values == np.int(start[0]))

        self.proj_stack = np.zeros([path_length, 48, 48])
        self._layer_x_vals = np.zeros(path_length)
        self.width = width

        for layer in np.arange(path_length):
            col = start_col - layer
            left_col = np.max([0, col-width])

            r_ix = (start_row * self.x_dim) + col
            l_ix = (start_row * self.x_dim) + left_col

            ix = np.arange(l_ix, r_ix) + 1

            if width > 1:
                self.proj_stack[layer] = np.mean(self.sysmat[ix], axis=0).reshape(48, 48)
            else:
                self.proj_stack[layer] = self.sysmat[ix].reshape(48, 48)

            self._layer_x_vals[layer] = self.x_values[col]

    def plot_total_sens_stack(self, y=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        layer_sums = self.proj_stack.sum(axis=(1, 2))
        ax.plot(self._layer_x_vals, layer_sums/layer_sums.max())
        ax.set_ylabel('Relative Sensitivity')
        ax.set_xlabel('X Position Right Edge (mm)')
        title = 'Sensitivity of {w} mm Width Lines'.format(w=int(self.width * self.pxl_size))
        if y:
            title += ' at y = {yp} mm'.format(yp=str(y))
        # ax.set_title('Sensitivity of {w} mm Width Lines'.format(w=int(self.width * self.pxl_size)))
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_proj(self, layer, show=True, save=True, save_name='proj'):
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'xticks': [], 'yticks': []})
        try:
            img = ax.imshow(self.proj_stack[layer], cmap='magma', origin='upper', interpolation='nearest')
        except Exception as e:
            print(e)
            img = ax.imshow(self.proj_stack[0], cmap='magma', origin='upper', interpolation='nearest')
        right = self._layer_x_vals[layer]
        ax.set_title('Projections from ({left} to {r}) mm'.format(left=right-self.width, r=right))
        fig.colorbar(img, ax=ax, fraction=0.045, pad=0.04)

        plt.tight_layout()
        if show:
            plt.show()

        if save:
            fig.savefig(save_name + '.png')


def main_projections(x_s=30, y_s=-10, width=10, steps=31, show=False, save=False):
    """projections saves projections, sens displays total sens plot, show shows each projection,
    save saves them in batch"""

    base_folder = '/Users/justinellin/Desktop/July_Work/current_sysmat/'
    sysmat_file = base_folder + '2021-07-03-1015_SP1.h5'
    display = sysmat_display(sysmat_file)

    start = (x_s, y_s)  # y_s = -10 is center
    display.generate_projection_stack(steps, start=start, width=width)

    base = '/Users/justinellin/repos/sysmat/july_basis/sysmat_tools/projections/'
    for p in np.arange(steps)[::1]:
        display.plot_proj(p, show=show, save=save, save_name=base + 'proj' + str(p))

    display.sysmat_file_obj.close()


def main_sens(x_s=30, y_s=-10, width=10, steps=31):
    """projections saves projections, sens displays total sens plot, show shows each projection,
    save saves them in batch"""

    base_folder = '/Users/justinellin/Desktop/July_Work/current_sysmat/'
    sysmat_file = base_folder + '2021-07-03-1015_SP1.h5'
    display = sysmat_display(sysmat_file)

    start = (x_s, y_s)  # y_s = -10 is center
    display.generate_projection_stack(steps, start=start, width=width)

    display.plot_total_sens_stack(y=y_s)
    display.sysmat_file_obj.close()


def main():
    x_s = 51
    steps = 103
    width = 10
    main_sens(x_s=x_s, steps=steps)


def main2():
    x_s = 50
    y_s = -10
    main_projections(x_s=x_s, y_s=y_s, width=1, steps=1, show=True, save=False)
    pass


if __name__ == "__main__":
    main2()

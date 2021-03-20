import numpy as np
import tables
# from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from processing.utils import compute_mlem
from processing.utils import interpolate_system_response
from processing.utils import load_h5file


def see_projection_together(sysmat_fname, choose_pt=0):
    sysmat_file = load_h5file(sysmat_fname)
    sysmat = sysmat_file.root.sysmat[:]

    plt.figure(figsize=(12, 8))
    plt.imshow(sysmat[choose_pt].reshape([48, 48]), cmap='jet', origin='lower', interpolation='nearest')
    sysmat_file.close()
    plt.show()


# def sensitivity_map(sysmat_fname, npix=(150, 50), pxl_sze= 1, dpix=(48, 48), correction=False):
def sensitivity_map(sysmat, npix=(150, 50), pxl_sze=1, dpix=(48, 48), correction=False):
    # sysmat_file = load_h5file(sysmat_fname)
    # sysmat = sysmat_file.root.sysmat[:]
    print("System Shape: ", sysmat.shape)

    sens = np.sum(sysmat, axis=1).reshape([npix[1], npix[0]])
    if correction:
        sens = np.mean(sens) / sens
    plt.figure(figsize=(12, 8))
    extent_img = np.array([-npix[0]/2, npix[0]/2, -npix[1]/2, npix[1]/2]) * pxl_sze
    img = plt.imshow(sens, cmap='jet', origin='lower', interpolation='nearest', aspect='equal', extent=extent_img)

    if correction:
        plt.title("Sensitivity Correction Map", fontsize=14)
    else:
        plt.title("Sensitivity Map", fontsize=14)
    plt.xlabel('[mm]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('[mm]', fontsize=14)

    plt.colorbar(img, fraction=0.046 * (sysmat.shape[0]/sysmat.shape[1]), pad=0.04)
    # sysmat_file.close()
    print("Total Sensitivity: ", np.sum(sysmat))
    print("Average Sensitivity: ", np.sum(sysmat)/np.prod(npix))
    plt.show()
    return sens


def see_projection_separate(sysmat_fname, choose_pt=0):
    sysmat_file = load_h5file(sysmat_fname)
    sysmat = sysmat_file.root.sysmat[:]

    point_response = sysmat[choose_pt].reshape([48, 48])

    layout = np.array([4, 4])

    fig, ax = plt.subplots(layout[0], layout[1])

    for plt_index in np.arange(layout[1] * layout[0]):
        row = plt_index // layout[1]
        col = plt_index % layout[0]

        data = point_response[(12 * row):(12 * (row+1)), (12 * col):(12 * (col+1))]

        im = ax[row, col].imshow(data, origin='lower')
        # plt.colorbar(im, ax=ax[row, col])
        ax[row, col].set_yticks([])
        ax[row, col].set_xticks([])

    fig.tight_layout()
    plt.show()


def test_mlem(sysmat_filename, h5file = True, check_proj=False, sensitivity_norm=True, point_check=None, flood=False,
              line_source=False, line_width=1, line_length=50, line_buffer=2, line_sigma=0.5,
              counts=10**6, img_pxl_x=75, img_pxl_y=25, pxl_sze=2, slice_plots=False, **kwargs):
    # img_pxl_y = 25
    # img_pxl_x = 75
    if point_check is None:
        point_check = img_pxl_x * img_pxl_y // 2

    test_img = np.zeros([img_pxl_y, img_pxl_x])

    if line_source:
        kern_shape = np.array([line_width, line_length + (2 * line_buffer)])  # line_width in y-dir, length in x-dir
        kern = np.zeros(kern_shape)
        kern[:, line_buffer:-line_buffer] = 1

        # print('Kernel: ', kern)

        mid_col = img_pxl_x//2 - ((img_pxl_x+1) % 2)
        mid_row = img_pxl_y//2 - ((img_pxl_y+1) % 2)

        col_offset = mid_col - (kern_shape[1]//2 - ((kern_shape[1]+1) % 2))
        row_offset = mid_row - (kern_shape[0]//2 - ((kern_shape[0]+1) % 2))

        test_img[row_offset:kern_shape[0] + row_offset, col_offset:kern_shape[1] + col_offset] = kern
        test_img = gaussian_filter(test_img, line_sigma, mode='constant')
        test_img = (test_img/test_img.sum() * counts)
           #  \ (kern/kern.sum() * counts)
    else:
        if flood:
            test_img.fill(counts//np.prod(test_img.shape))  # flood source
        else:
            test_img[np.unravel_index(point_check, test_img.shape)] = counts  # point source

    # plt.imshow(test_img,  cmap='jet', origin='lower')
    # plt.show()
    if h5file:
        sysmat_file = load_h5file(sysmat_filename)
        sysmat = sysmat_file.root.sysmat[:].T
    else:
        sysmat = np.load(sysmat_filename).T

    test_counts = sysmat.dot(test_img.ravel()).round()
    if sensitivity_norm:
        sens = np.sum(sysmat, axis=0)
    else:
        sens = np.ones(sysmat.shape[1])

    test_counts = np.random.poisson(test_counts)  # TODO: Remove if finding local PSFs
    recon = compute_mlem(sysmat, test_counts, img_pxl_x, sensitivity=sens, **kwargs)

    if check_proj:
        fig, ax = plt.subplots(4, 4)

        point_response = test_counts.reshape([48, 48])
        for plt_index in np.arange(4 * 4):
            row = plt_index // 4
            col = plt_index % 4

            data = point_response[(12 * row):(12 * (row+1)), (12 * col):(12 * (col+1))]

            im = ax[row, col].imshow(data, origin='lower')
            # plt.colorbar(im, ax=ax[row, col])
            ax[row, col].set_yticks([])
            ax[row, col].set_xticks([])

        # sysmat_file.close()
        fig.tight_layout()
        plt.show()

    plt.figure(figsize=(12, 8))
    extent_img = np.array([-img_pxl_x / 2, img_pxl_x / 2, -img_pxl_y / 2, img_pxl_y / 2]) * pxl_sze
    img = plt.imshow(recon.reshape([img_pxl_y, img_pxl_x]), cmap='jet', origin='lower', interpolation='nearest',
                     extent=extent_img)

    xcoords = [0, 25, 50]
    colors = ['k', 'r', 'm']
    # for xc, c in zip(xcoords, colors):
    #    plt.axvline(x=xc, linestyle='--', label='line at x = {}'.format(xc), c=c, linewidth=2)
    # plt.axhline(y=0, linestyle='--', c=colors[0], linewidth=2)

    plt.xlabel('[mm]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('[mm]', fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Total Counts: " + np.format_float_scientific(int(recon.sum()), precision=2), fontsize=20)

    print("Total Counts: ", recon.sum())

    if h5file:
        sysmat_file.close()

    plt.colorbar(img, fraction=0.046 * (sysmat.shape[0]/sysmat.shape[1]), pad=0.04)
    plt.show()

    if not slice_plots:
        return

    # (np.ceil(img_pxl_y) // 2 * 2) + 1  # round to nearest odd integer
    x0_plane_idx = int(np.ceil(img_pxl_x / 2))
    y0_plane_idx = int(np.ceil(img_pxl_y / 2))

    thirds = (img_pxl_x - x0_plane_idx)//3
    x_planes_interest_idx = [x0_plane_idx + (thirds * shift) for shift in np.arange(3)]

    plt.figure(figsize=(12, 8))
    x_vals = np.linspace(extent_img[0], extent_img[1], img_pxl_x)
    plt.plot(x_vals, recon.reshape([img_pxl_y, img_pxl_x])[y0_plane_idx, :], label='Single Slice')
    plt.xlabel('[mm]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.yticks(fontsize=14)
    # plt.plot(x_vals, np.mean(recon.reshape([img_pxl_y, img_pxl_x])[13-1:13+1+1, :], axis=0), label='3 Row Average')
    plt.title("Central (y=0) Slice")
    plt.show()

    plt.figure(figsize=(12, 8))
    y_vals = np.linspace(extent_img[2], extent_img[3], img_pxl_y)
    for (plane_id, color, plane_label) in zip(x_planes_interest_idx, colors, xcoords):
        plt.plot(y_vals, recon.reshape([img_pxl_y, img_pxl_x])[:, plane_id],
                 label='x = {} mm'.format(plane_label), c=color)
    plt.xlabel('[mm]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.yticks(fontsize=14)
    # plt.plot(x_vals, np.mean(recon.reshape([img_pxl_y, img_pxl_x])[13-1:13+1+1, :], axis=0), label='3 Row Average')
    plt.title("Vertical Slices")
    plt.legend()
    plt.show()

    print("y0_plane_idx: ", y0_plane_idx)
    np.save('central_slice', recon.reshape([img_pxl_y, img_pxl_x]))


def system_matrix_interpolate(sysmat_filename, save=False):
    sysmat_file = load_h5file(sysmat_filename)
    sysmat = sysmat_file.root.sysmat[:]
    if save:
        save_name = sysmat_filename[:-3] + '_interp'
        interpolate_system_response(sysmat, x_img_pixels=75, save_fname=save_name)
    else:
        interpolate_system_response(sysmat, x_img_pixels=75)


def main():
    # see_projection_together(filename, choose_pt=937)
    # see_projection_separate(filename, choose_pt=937)

    # filename = '/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0_interp.npy'
    filename = '/home/justin/repos/python3316/processing/2021-02-28-2345_SP0_F1S7.npy'
    data_file = '/home/justin/Desktop/images/zoom_fixed/thor10_07.npz'
    sysmat = np.load(filename)
    # sysmat_file = load_h5file(filename)
    # sysmat = sysmat_file.root.sysmat[:]
    # 149, 49 interp size
    sensitivity_map(sysmat, npix=(149, 49), pxl_sze=1, correction=False)


if __name__ == '__main__':
    # main()

    # fname = '/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0_interp.npy'

    fname = '/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0_F1S7.npy'
    # sysmat = np.load(fname)
    # sens_correct = sensitivity_map(sysmat, npix=(149, 49), pxl_sze=1, correction=True)
    # test_mlem(sysmat_filename=fname,
    #          line_source=True, line_length=100, line_buffer=4, line_sigma=1, line_width=1, filt_sigma=[0.5, 4.5],
    #          img_pxl_x=149, img_pxl_y=49, pxl_sze=1, counts=10**8, slice_plots=True,
    #          nIterations=800, h5file=False)  # TODO: Generate and give to Josh? Interpolate and non-interpolated.

    test_mlem(sysmat_filename='/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0.h5',
              line_source=True, filt_sigma=[0.25, 1], nIterations=500, counts=10**8, slice_plots=True)
    # [0.25, 1] for nice thin line source, [0.5, 1] wide source
    # test_mlem(sysmat_filename='/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0.h5',
    #          line_source=False, filt_sigma=0.5, nIterations=100)  # Flood test

    # system_matrix_interpolate('/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0.h5', save=True)

import numpy as np
import matplotlib.pyplot as plt


def flip_det(proj_array, ind, flip_ud=False, n_rot=1, ndets=(4, 4), det_pxls=(12, 12)):
    """flip_ud flips up/down. n_rot is number of 90 degree rotations, ndets = (row,col), det_pxls = ny, nx
    ind starts at 0 in upper left, to 3 in upper right, left to right up to down when facing front of det.
    proj_array is loaded proj_array. Flip happens before rotation"""
    det_rows, det_cols = ndets
    row = ind //det_rows  # 0 is top row
    col = ind % det_cols  # 0 is on left
    ny, nx = det_pxls

    proj = np.copy(proj_array).reshape([ny * det_rows, nx * det_cols])
    # area = proj[(col * ny):((col + 1) * ny), (row * nx):((row + 1) * nx)]
    area = proj[(row * nx):((row + 1) * nx), (col * ny):((col + 1) * ny)]

    if flip_ud:
        area = area[::-1]

    # proj[(col * ny):((col+1)*ny), (row * nx):((row+1)*nx)] = np.rot90(area, n_rot)
    proj[(row * nx):((row + 1) * nx), (col * ny):((col + 1) * ny)] = np.rot90(area, n_rot)
    return proj


def weights(mid_include=True):

    mid_wgt = 1
    edge_wgts = 1/4 * mid_wgt
    corner_wgts = 1/2 * edge_wgts
    save_name = 'det_correction_mid'

    if not mid_include:
        mid_wgt = 0
        edge_wgts = 1
        corner_wgts = 1/2
        save_name = 'det_correction_no_mid'

    interior_pxl = mid_wgt + (4 * edge_wgts) + (4 * corner_wgts)
    edge_pxls = mid_wgt + (3 * edge_wgts) + (2 * corner_wgts)
    corner_pxls = mid_wgt + (2 * edge_wgts) + (1 * corner_wgts)

    edge_gain_correction = interior_pxl / edge_pxls
    corner_gain_correction = interior_pxl / corner_pxls
    return edge_gain_correction, corner_gain_correction, save_name


def test_flip(filename, mod, flip=False, rotations=1, **kwargs):
    # file = '/Users/justinellin/Dissertation/August/carbon_scatter/pos29mm_Aug1.npz'

    data = np.load(filename)
    orig_img = data['image_list']
    new_img = flip_det(orig_img, mod, flip_ud=flip, n_rot=rotations, **kwargs)
    # print("Data keys: ", [*data])
    # print("Data['image_list']: ", data['image_list'])

    titles = ('original', 'mod flipped')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 5),
                            subplot_kw={'xticks': [], 'yticks': []})

    for ax, proj, title in zip(axs, (orig_img, new_img), titles):
        img = ax.imshow(proj, cmap='magma', origin='upper', interpolation='nearest')
        ax.set_title(title)
        # fig.colorbar(img, ax=ax, fraction=0.045, pad=0.04)

    plt.tight_layout()
    plt.show()


def main(save=True, **kwargs):
    ndets = np.array((4, 4))
    det_template =  np.ones([12, 12])
    print(weights(**kwargs))
    egc, cgc, save_name = weights(**kwargs)  # edge_gain_correction, corner_gain_correction

    det_template[0] = egc
    det_template[-1] = egc
    det_template[:, 0] = egc
    det_template[:, -1] = egc

    cidx = np.ix_((0, -1), (0, -1))
    det_template[cidx] = cgc

    correction = np.tile(det_template, ndets)
    # print("det_template: ", det_template)
    if save:
        np.save(save_name, correction)


if __name__ == "__main__":
    # main(mid_include=False, save=False)
    test_flip('/Users/justinellin/Dissertation/August/carbon_scatter/pos29mm_Aug1.npz', 11,
              flip=True, rotations=0)

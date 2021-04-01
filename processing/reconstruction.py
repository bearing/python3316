import numpy as np
import time
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
# from scipy.stats import linregress, moment
import tables


def compute_mlem_full(sysmat, counts, dims,  # env_dims, shield_dims,
                      x_det_pixels=48,
                      sensitivity=None,
                      det_correction=None,
                      nIterations=10,
                      filter='gaussian',
                      filt_sigma=1,
                      **kwargs):
    """Major difference is that this will also reconstruct response for environment. img and env dims in (x, y))"""
    print("Total Measured Counts: ", counts.sum())  # TODO: Normalize?

    tot_det_pixels, tot_img_pixels = sysmat.shape  # n_measurements, n_pixels
    regions = [' Object ', ' Table ', ' Shielding ']

    tot_reg_pxls = [None] * len(dims)

    tot_obj_plane_pxls = dims[0].prod()
    tot_reg_pxls = [tot_obj_plane_pxls]

    x_obj_pixels, y_obj_pixels = dims[0]

    reg_ids = np.arange(len(dims))

    print("Total Detector Pixels: ", tot_det_pixels)
    for r in reg_ids:
        print("Total", regions[r], 'Pixels: ', np.prod(dims[r]))
        tot_reg_pxls.append(np.prod(dims[r]))
    print("Total Image Pixels: ", tot_img_pixels)

    if sensitivity is None:
        sensitivity = np.ones(tot_img_pixels)

    if det_correction is None:
        det_correction = np.ones(tot_det_pixels)

    sensitivity = sensitivity.ravel()
    det_correction = det_correction.ravel()

    measured = counts.ravel() * det_correction

    if nIterations == 1:
        return sysmat.T.dot(measured)/sensitivity  # Backproject

    recon_img = np.ones(tot_img_pixels)
    recon_img_previous = np.ones(recon_img.shape)
    # diff = 10**6 * np.ones(recon_img.shape)
    # outSum = np.zeros_like(recon_img)

    if sensitivity is None:
        sensitivity = np.ones(tot_img_pixels)

    itrs = 0
    t1 = time.time()

    while itrs < nIterations:  # and (diff.sum() > 0.001 * counts.sum() + 100):
        sumKlamb = sysmat.dot(recon_img)
        outSum = (sysmat * measured[:, np.newaxis]).T.dot(1/sumKlamb)
        recon_img *= outSum / sensitivity

        if itrs > 5 and filter == 'gaussian':
            recon_img[:tot_obj_plane_pxls] = \
                gaussian_filter(recon_img[:tot_obj_plane_pxls].reshape([y_obj_pixels, x_obj_pixels]),
                                        filt_sigma, **kwargs).ravel()
            # gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

        print('Iteration %d, time: %f sec' % (itrs, time.time() - t1))
        diff = np.abs(recon_img - recon_img_previous)
        print('Diff Sum: ', diff.sum())
        recon_img_previous[:] = recon_img
        itrs += 1
    print("Total Iterations: ", itrs)

    inds = np.cumsum(tot_reg_pxls)[:-1]  # for split function
    recons = np.split(recon_img, inds)  # these are raveled

    for r in reg_ids:
        recons[r] = recons[r].reshape(dims[r][::-1])

    return recons  # obj, table, shielding


def load_h5file(filepath):  # h5file.root.sysmat[:]
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def sensivity_map(sysmat_fname, npix=(150, 50), dpix=(48, 48)):
    sysmat_file = load_h5file(sysmat_fname)
    sysmat = sysmat_file.root.sysmat[:]
    print("Sysmat shape: ", sysmat.shape)

    sens = np.sum(sysmat, axis=1).reshape([npix[1], npix[0]])
    print("Total Sensitivity: ", np.sum(sysmat))
    print("Average Sensitivity: ", np.sum(sysmat)/np.prod(npix))
    plt.figure(figsize=(12, 8))

    extent_img = [-npix[0]/2, npix[0]/2, -npix[1]/2, npix[1]/2]
    img = plt.imshow(sens, cmap='jet', origin='upper', interpolation='nearest', aspect='equal', extent=extent_img)
    plt.title("Sensitivity Map", fontsize=14)
    plt.xlabel('mm', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('mm', fontsize=14)
    plt.yticks(fontsize=14)

    plt.colorbar(img, fraction=0.046 * (sysmat.shape[0]/sysmat.shape[1]), pad=0.04)
    plt.show()
    sysmat_file.close()


def see_projection(sysmat_fname, choose_pt=0, npix=(150, 50), dpix=(48, 48)):
    sysmat_file = load_h5file(sysmat_fname)
    sysmat = sysmat_file.root.sysmat[:]
    print("Sysmat shape: ", sysmat.shape)

    sens = np.sum(sysmat, axis=1).reshape([npix[1], npix[0]])
    print("Total Sensitivity: ", np.sum(sysmat))
    print("Average Sensitivity: ", np.sum(sysmat)/np.prod(npix))
    plt.figure(figsize=(12, 8))

    # extent_img = [-npix[0]/2, npix[0]/2, -npix[1]/2, npix[1]/2]
    img = plt.imshow(sysmat[choose_pt].reshape([48, 48]), cmap='magma', origin='upper', interpolation='nearest', aspect='equal')
    plt.title("Projection", fontsize=14)
    # plt.xlabel('mm', fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.ylabel('mm', fontsize=14)
    # plt.yticks(fontsize=14)

    plt.colorbar(img, fraction=0.046 * (sysmat.shape[0]/sysmat.shape[1]), pad=0.04)
    plt.show()
    sysmat_file.close()


def image_reconstruction_full(sysmat_file, data_file,
                              obj_pxls, env_pxls=(0, 0), shield_pxls=(0, 0),
                              obj_center=(0, 0), env_center=(0, - 130),  # shield_center=(-150.49, -33.85, -168.89),
                              pxl_sze=(1, 10), **kwargs):
    """For use with compute_mlem2. Generates two images. Object plane and environment"""
    from matplotlib.gridspec import GridSpec

    plot_titles = ['Object FOV', 'Table', 'Shielding']
    labels_x = ['Beam [mm]', 'Beam [mm]', 'Horizontal [mm]']
    labels_y = ['Vertical [mm]', 'System Axis [mm]', 'Vertical [mm]']
    filename = sysmat_file

    if tables.is_hdf5_file(filename):
        sysmat_file_obj = load_h5file(filename)
        sysmat = sysmat_file_obj.root.sysmat[:].T
    else:
        sysmat = np.load(filename).T
    data = np.load(data_file)
    counts = data['image_list']

    dims = [np.array(obj_pxls)]
    centers = [obj_center]

    if env_pxls != (0, 0):
        dims.append(np.array(env_pxls))
        centers.append(env_center)
    if shield_pxls != (0, 0):
        dims.append(np.array(shield_pxls))
        centers.append((0, 0))  # TODO: Figure out how to plot this

    plots = len(dims)
    print("Obj_pxls: ", obj_pxls)

    recons = compute_mlem_full(sysmat, counts, dims,
                               sensitivity=np.sum(sysmat, axis=0), **kwargs)  # TODO: Variable outputs
    # obj_recon, table_recon, shielding_recon

    fig = plt.figure(figsize=(4 * plots, 3 * plots))
    gs = GridSpec(1, plots)

    axes = [None] * plots
    params = axes.copy()

    for p in np.arange(plots):
        ax = fig.add_subplot(gs[0, p])
        pxl = pxl_sze[p]
        extent_x = centers[p][0] + (np.array([-1, 1]) * dims[p][0] / 2 * pxl)
        extent_y = centers[p][1] + (np.array([-1, 1]) * dims[p][1] / 2 * pxl)
        params[p] = [recons[p], np.arange(extent_x[0], extent_x[1], pxl_sze[p]) + 0.5 * pxl,
                     np.arange(extent_y[0], extent_y[1], pxl_sze[p]) + 0.5 * pxl]

        img = ax.imshow(recons[p], cmap='magma', origin='upper', interpolation='nearest',
                        extent=np.append(extent_x, extent_y))

        ax.set_title(plot_titles[p])
        ax.set_xlabel(labels_x[p])
        ax.set_ylabel(labels_y[p])
        fig.colorbar(img, fraction=0.046, pad=0.04, ax=ax)

    fig.tight_layout()

    plt.show()
    return params


if __name__ == "__main__":
    # see_projection('/home/proton/repos/python3316/processing/system_responses/2021-02-23-1514_SP0.h5',
    #               choose_pt=950, npix=npix)

    # npix = np.array([149, 49])  # March 17 present
    # npix = np.array([101, 21])  # new offcenter system response
    # npix = np.array([201, 49])  # 3/23 centered Davis
    # center = (0, -25)  # 3/23 centered Davis
    # npix = np.array([201, 49 + 53])  # combined (appended)
    # center = (0, -50.5)  # combined
    npix = np.array([201, 151])
    center = (0, 0)
    # see_projection('/home/justin/repos/sysmat/design/2021-03-18-2312_SP0.h5', choose_pt=1060, npix=npix)
    data_file = '/home/justin/Desktop/images/zoom_fixed/thor10_07.npz'  # overnight
    data_0cm = '/home/justin/Desktop/images/recon/thick07/0cm.npz'
    data_6cm = '/home/justin/Desktop/images/recon/thick07/6cm.npz'
    data_12cm = '/home/justin/Desktop/images/recon/thick07/12cm.npz'

    # ====Image Recon 2 ====
    center_env = (0, -110)  # (x, z)
    sysmat_fname = '/home/justin/repos/python3316/processing/3_31_2021_tot.npy'
    obj_params, env_params = image_reconstruction_full(sysmat_fname, data_6cm,
                                                   obj_pxls=npix,
                                                   env_pxls=(21, 23),
                                                   obj_center=center,
                                                   env_center=center_env,
                                                   filt_sigma=[0.25, 1],
                                                   nIterations=30)
    # TODO: Modify this so it dynammically plots images of each region based on given system responses and envir.
    # TODO: Put flags so it automatically interpolates and smooths raw responses
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

    # x_obj_pixels, y_obj_pixels = dims[0]  # TODO: original
    try:
        x_obj_pixels, y_obj_pixels, zlayers = dims[0]
    except Exception as e:
        x_obj_pixels, y_obj_pixels = dims[0]
        zlayers = 1

    reg_ids = np.arange(len(dims))

    # print("Z layers: ", zlayers)
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

    # if nIterations == 1:
    #    return sysmat.T.dot(measured)/sensitivity  # Backproject

    recon_img = np.ones(tot_img_pixels)
    recon_img_previous = np.ones(recon_img.shape)

    if sensitivity is None:
        sensitivity = np.ones(tot_img_pixels)

    itrs = 0
    t1 = time.time()

    z_indices = tot_obj_plane_pxls // zlayers * (np.arange(zlayers) + 1)  # each z-layer
    print("Z indices: ", z_indices)
    tmp_store = [None] * 4

    # check = 1

    while itrs < nIterations:  # and (diff.sum() > 0.001 * counts.sum() + 100):
        sumKlamb = sysmat.dot(recon_img)
        outSum = (sysmat * measured[:, np.newaxis]).T.dot(1/sumKlamb)
        recon_img *= outSum / sensitivity

        if itrs > 5 and filter == 'gaussian':
            layers = np.split(recon_img[:tot_obj_plane_pxls], z_indices[:-1])
            for zid, layer in enumerate(layers):
                tmp_store[zid] = gaussian_filter(layer.reshape([y_obj_pixels, x_obj_pixels]),
                                              filt_sigma, **kwargs).ravel()
                # if check and itrs == 29:
                #     plt.imshow(tmp_store[zid].reshape([y_obj_pixels, x_obj_pixels]),
                #                cmap='magma', origin='upper', interpolation='nearest')
                #     plt.show()
            recon_img[:tot_obj_plane_pxls] = np.concatenate(tmp_store)
            # check = 0
            # if check and itrs == 29:
            #     check = 0
        print('Iteration %d, time: %f sec' % (itrs, time.time() - t1))
        diff = np.abs(recon_img - recon_img_previous)
        print('Diff Sum: ', diff.sum())
        recon_img_previous[:] = recon_img
        itrs += 1
    print("Total Iterations: ", itrs)

    inds = np.cumsum(tot_reg_pxls)[:-1]  # for split function
    recons = np.split(recon_img, inds)  # these are raveled

    for r in reg_ids:
        # print("R: ", r)
        # print("recons[r]: ", recons[r].shape)
        if len(dims[r]) == 2:
            recons[r] = recons[r].reshape(dims[r][::-1])
        else:
            recons[r] = recons[r].reshape(dims[r][[2, 1, 0]])

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

    try:
        niter = kwargs['nIterations']
    except Exception as e:
        niter = 10

    try:
        kern = kwargs['filt_sigma']
    except Exception as e:
        kern = 1

    plot_titles = ['Object FOV ({n} Iterations, kernel: {k})'.format(n=niter, k=kern), 'Table', 'Shielding']
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
                               sensitivity=np.sum(sysmat, axis=0), **kwargs)
    # obj_recon, table_recon, shielding_recon

    fig = plt.figure(figsize=(12, 9))
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

        tx = ax.set_title(plot_titles[p])
        ax.set_xlabel(labels_x[p])
        ax.set_ylabel(labels_y[p])

        try:
            img = ax.imshow(recons[p][0, ...], cmap='magma', origin='upper', interpolation='nearest',
                            extent=np.append(extent_x, extent_y))
        except Exception as e:
            img = ax.imshow(recons[p], cmap='magma', origin='upper', interpolation='nearest',
                            extent=np.append(extent_x, extent_y))
        fig.colorbar(img, fraction=0.046, pad=0.04, ax=ax)

        # if dims[p].size == 3:
        #    for zid in np.arange(1, dims[p][2]):
        #        img.set_data(recons[p][..., zid])
        #        fig.canvas.draw_idle()
        #        plt.pause(1)

    fig.tight_layout()

    plt.show()
    return params


if __name__ == "__main__":
    npix = np.array([37, 31, 4])
    center = (-12, -10)

    data_file = '/home/justin/Desktop/images/zoom_fixed/thor10_07.npz'  # overnight
    data_0cm = '/home/justin/Desktop/images/recon/thick07/0cm.npz'
    data_6cm = '/home/justin/Desktop/images/recon/thick07/6cm.npz'
    data_12cm = '/home/justin/Desktop/images/recon/thick07/12cm.npz'

    # sysmat_fname = '/home/justin/repos/python3316/processing/4_2_processed_F1S7.npy'  # 3d obj
    sysmat_fname = '/home/justin/repos/python3316/processing/fov3d.npy'
    iterations = 60
    params = image_reconstruction_full(sysmat_fname, data_6cm,
                                                       npix,  # obj_pxls
                                                       # env_pxls=(21, 23),  # tot
                                                       obj_center=center,
                                                       # env_center=center_env,  # tot
                                                       filt_sigma=[0.25, 0.5],  # vertical, horizontal
                                                       nIterations=iterations)

    obj_img, x_range, y_range = params[0]
    extent_x = np.array([x_range[0], x_range[-1]])
    extent_y = np.array([y_range[0], y_range[-1]])

    print("Extent 2: ", np.append(extent_x, extent_y))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    min = obj_img.min()
    max = obj_img.max()
    z_dist = ['120', '110', '100', '90']
    for z_ind, img_z in enumerate(obj_img):
        ax = axes.flat[z_ind]
        im = ax.imshow(img_z, cmap='magma', origin='upper', interpolation='nearest',
                       extent=np.append(extent_x, extent_y), vmin=min, vmax=max)
        # im = ax.imshow(img_z, cmap='magma', origin='upper', interpolation='nearest',
        #                extent=np.append(extent_x, extent_y))  # Colorbars normalized only to layer
        ax.set_title('Z (dist to collimator) = {z} mm '.format(z=z_dist[z_ind]))
        ax.set_xlabel('[mm]')
        ax.set_ylabel('[mm]')

    # fig.tight_layout()
    fig.suptitle("Pos. 6 (center)", fontsize=24)
    # === Normalized ===
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    # === Normalized ===
    plt.show()
    # plt.plot(obj_params[1], np.sum(obj_params[0], axis=0))
    # plt.title("Projection Along Beam ({n} iterations)".format(n=iterations))
    # plt.xlabel("Distance [mm]")
    # plt.show()
    # TODO: Put flags so it automatically interpolates and smooths raw responses
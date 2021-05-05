import numpy as np
import time
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
# from scipy.stats import linregress, moment
import tables
import sys


def compute_mlem_full(sysmat, counts, dims,  # env_dims, shield_dims,
                      x_det_pixels=48,
                      sensitivity=None,
                      det_correction=None,
                      nIterations=10,
                      filter='gaussian',
                      filt_sigma=1,
                      **kwargs):
    """Major difference is that this will also reconstruct response for environment. img and env dims in (x, y))"""
    print("Total Measured Counts: ", counts.sum())

    tot_det_pixels, tot_img_pixels = sysmat.shape  # n_measurements, n_pixels
    regions = [' Object ', ' Region 1 ', ' Region 2 ']  # TODO: Dynamically expand this list

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

    # if nIterations == 1:
    #    return sysmat.T.dot(measured)/sensitivity  # Backproject

    recon_img = np.ones(tot_img_pixels)
    recon_img_previous = np.ones(recon_img.shape)
    diff = np.ones(recon_img.shape)

    if sensitivity is None:
        sensitivity = np.ones(tot_img_pixels)

    itrs = 0
    t1 = time.time()

    # TODO: Machine errors of low values, check that this is necessary
    # sysmat[sysmat == 0] = np.mean(sysmat) * 0.001  # Original
    sysmat[sysmat == 0] = np.min(sysmat[sysmat != 0])

    while itrs < nIterations and (diff.sum() > 0.001 * counts.sum() + 100):
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
        print("R: ", r)
        print("recons[r]: ", recons[r].shape)
        recons[r] = recons[r].reshape(dims[r][::-1])

    return recons  # obj, region 1, region 2, region 3, etc.


def load_h5file(filepath):  # h5file.root.sysmat[:]
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def load_sysmat(sysmat_fname):
    if tables.is_hdf5_file(sysmat_fname):
        sysmat_file_obj = load_h5file(sysmat_fname)
        return sysmat_file_obj.root.sysmat[:].T
    return np.load(sysmat_fname).T


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

    if tables.is_hdf5_file(sysmat_fname):
        sysmat_file = load_h5file(sysmat_fname)
        sysmat = sysmat_file.root.sysmat[:]
    else:
        sysmat = np.load(sysmat_fname)

    # sysmat_file = load_h5file(sysmat_fname)
    # sysmat = sysmat_file.root.sysmat[:]
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
                              pxl_sze=(2, 10), edge_correction=False, sides_interp=False,
                              batch_fname=None, show_plot=True,
                              **kwargs):
    """For use with compute_mlem_full. Generates two images. Object plane and environment"""
    from matplotlib.gridspec import GridSpec
    from utils import edge_gain

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

    if edge_correction:  # TODO: Test, probably should remove
        counts = edge_gain(counts, sides_interp=sides_interp)  # kwarg -> sides_interp

    print("Sysmat shape: ", sysmat.shape)
    # assert np.prod(np.array(obj_pxls)) == sysmat.shape[1], \
    #    "Mismatch between obj dims, {o}, and response: {r}".format(o=np.array(obj_pxls), r=sysmat.shape)

    dims = [np.array(obj_pxls)]
    centers = [obj_center]

    if env_pxls != (0, 0):
        dims.append(np.array(env_pxls))
        centers.append(env_center)
    if shield_pxls != (0, 0):
        dims.append(np.array(shield_pxls))
        centers.append((0, 0))  # TODO: Figure out how to plot this

    if len(dims) == 1:
        assert np.prod(np.array(obj_pxls)) == sysmat.shape[1], \
            "Mismatch between obj dims, {o}, and response: {r}".format(o=np.array(obj_pxls), r=sysmat.shape)
    else:
        assert np.prod(np.array(dims), axis=1).sum() == sysmat.shape[1], \
            "Mismatch between total dims, {o}, and response: {r}".format(o=np.array(dims), r=sysmat.shape)

    plots = len(dims)
    print("Obj_pxls: ", obj_pxls)

    recons = compute_mlem_full(sysmat, counts, dims,
                               sensitivity=np.sum(sysmat, axis=0), **kwargs)  # TODO: Variable outputs
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

        img = ax.imshow(recons[p], cmap='magma', origin='upper', interpolation='nearest',
                        extent=np.append(extent_x, extent_y))

        ax.set_title(plot_titles[p])
        ax.set_xlabel(labels_x[p])
        ax.set_ylabel(labels_y[p])
        fig.colorbar(img, fraction=0.046, pad=0.04, ax=ax)

    fig.tight_layout()

    if show_plot:
        plt.show()
    if type(batch_fname) is str:
        plt.savefig(batch_fname, bbox_inches="tight")
    plt.close(fig)  # TODO: Is this necessary?
    return params


# ===========Batch Versions===========
class Reconstruction(object):
    def __init__(self, sysmat_filename, region_pxls, region_centers, pxl_sizes):
        n_regions, reg_dims = region_pxls.shape
        assert reg_dims == 2, "Expected (n, 2) shape for region_pxls. Got {s} instead.".format(s=region_pxls.shape)
        self.n_regions = n_regions
        if pxl_sizes.size == 1 and self.n_regions > 1:
            self.pxl_size = np.repeat(pxl_sizes, self.n_regions)
        else:
            self.pxl_sizes = pxl_sizes
        self.region_dims = region_pxls
        self.region_centers = region_centers

        # Needed for plot limits
        self.figure, self.axes, self.imgs, self.cbars = self.initialize_figures()
        self.line_projections = np.zeros([1, region_pxls[0, 1]])  # first must be object FoV

        self.sysmat = self.load_sysmat_from_file(sysmat_filename)
        self.recons = [None] * self.n_regions

    def initialize_figures(self):
        x_labels = ['Beam [mm]']
        y_labels = ['Vertical [mm]']
        for i in np.arange(1, self.n_regions):
            x_labels.append('R' + str(i) + ' axis 0 [mm]')
            y_labels.append('R' + str(i) + ' axis 1 [mm]')

        extent_x = self.region_centers[:, 0][:, np.newaxis] + \
                        (np.array([-1, 1]) * (self.region_dims[:, 0] * self.pxl_sizes)[:, np.newaxis])/2
        extent_y = self.region_centers[:, 1][:, np.newaxis] + \
                        (np.array([-1, 1]) * (self.region_dims[:, 1] * self.pxl_sizes)[:, np.newaxis])/2

        fig = plt.figure(figsize=(12, 9), constrained_layout=False)
        cols = 3  # hardcoded for now
        rows = int(np.ceil(self.n_regions / cols))
        gs = fig.add_gridspec(nrows=rows, ncols=cols)

        axes_objs = []
        img_objs = []
        cbars = []

        for id, (r_dims, x_label, y_label, rng_x, rng_y) in \
                enumerate(zip(self.region_dims, x_labels, y_labels, extent_x, extent_y)):
            row = id // cols
            col = id % cols

            ax = fig.add_subplot(gs[row, col])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if id != 0:  # TODO: First needs to list iters and filter
                ax.set_title('Region ' + str(id) + ' Image')
            img = ax.imshow(np.ones(r_dims[::-1]), cmap='magma', origin='upper',
                            interpolation='nearest', extent=np.append(rng_x, rng_y))

            cbars.append(fig.colorbar(img, fraction=0.046, pad=0.04, ax=ax))

            axes_objs.append(ax)
            img_objs.append(img)
        fig.tight_layout()
        return fig, axes_objs, img_objs, cbars

    def mlem_reconstruct(self, data_file, recon_save_fname=None, **kwargs):

        try:
            niter = kwargs['nIterations']
        except Exception as e:
            niter = 10

        try:
            kern = kwargs['filt_sigma']
        except Exception as e:
            kern = 1
        self.axes[0].set_title('Object FOV ({n} Iterations, kernel: {k})'.format(n=niter, k=kern))
        # First axis is always of the obj FoV (and not background)

        data = np.load(data_file)
        counts = data['image_list']
        self.recons = compute_mlem_full(self.sysmat, counts, self.region_dims,
                                        sensitivity=np.sum(self.sysmat, axis=0), **kwargs)
        # return recons  # TODO: Care that you don't need a copy()

    def update_plots(self, norm_plot=False, show_plot=False):
        min, max = self.global_limits(self.recons)

        for region_image, recon, cbar in zip(self.imgs, self.recons, self.cbars):
            region_image.set_data(recon)

            if norm_plot:  # normalize to global min/max
                cbar.set_clim(vmin=min, vmax=max)
            else:  # normalize to RoI min/max
                cbar.set_clim(vmin=recon.min(), vmax=recon.max())
            cbar.draw_all()
            # plt.draw()
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        if show_plot:
            plt.show()

        # TODO: Break up this function into just a plotter/updater and saver. Return recons and save

    def global_limits(self, recons):
        min = np.inf
        max = 0
        for region in recons:
            if region.min() < min:
                min = region.min()
            if region.max() > max:
                max = region.max()
        return min, max

    def load_sysmat_from_file(self, filename):
        sysmat = load_sysmat(filename)
        total_expected_pxls = np.product(self.region_dims.T, axis=0).sum()
        assert total_expected_pxls == self.sysmat.shape[1], \
            "Mismatch between expected pixels, {o}, and response: {r}".format(o=total_expected_pxls,
                                                                              r=self.sysmat.shape)
        return sysmat


def split_image_and_project(sysmat_file, data_file, recon_image, section_width_x=None, batch=False, norm=True):
    """sysmat_file = system response, data_file = projection, recon_image = first params from full recon,
    section width x is length in pixels of sections of image. If none provided defaults to full"""
    from matplotlib.gridspec import GridSpec
    sysmat = load_sysmat(sysmat_file)[:, :recon_image.size]  # system response  # TODO: did this fix it?

    data = np.load(data_file)
    counts = data['image_list']  # measured counts
    p_limits = [counts.min(), counts.max()]  # original colorbar limits for projection
    i_limits = [recon_image.min(), recon_image.max()]  # original image recon colorbar limits

    ip_y, ip_x = recon_image.shape  # ip = image pixel
    # print("Recon_image shape: ", recon_image.shape)

    if section_width_x is None:
        section_width_x = ip_x

    front_idx = np.cumsum(section_width_x)
    back_idx = np.r_[0, front_idx[:-1]]

    # masked_image = np.zeros(recon_image.shape)
    x_ind = np.arange(ip_x)

    sections = len(section_width_x)

    # fig = plt.figure(figsize=(16, 12))
    fig = plt.figure(figsize=(16, 12), clear=True)  # TODO: Attempt at Fix
    gs = GridSpec(2, sections + 1)

    # axes = [None] * plots
    ax0 = fig.add_subplot(gs[0, -1])  # original data
    ax1 = fig.add_subplot(gs[1, -1])  # full image recon
    plot_titles = ["Measured Projection", "Full Image Recon"]

    for title, ax, image, climits in zip(plot_titles, [ax0, ax1], [counts, recon_image], [p_limits, i_limits]):
        img = ax.imshow(image, cmap='magma', origin='upper', interpolation='nearest', vmin=climits[0], vmax=climits[1])
        fig.colorbar(img, fraction=0.046, pad=0.04, ax=ax)
        ax.set_title(title)

    for sid, [bid, fid] in enumerate(zip(back_idx, front_idx)):
        masked_image = recon_image * ((bid-1 < x_ind) * (x_ind < fid))
        print("Sysmat.shape: ", sysmat.shape)
        print("Masked image shape: ", masked_image.shape)
        forward_project = (sysmat @ masked_image.ravel()).reshape([48, 48])
        ax_p = fig.add_subplot(gs[0, sid])
        ax_i = fig.add_subplot(gs[1, sid])
        print("Percent of Recon Counts in Section: ", masked_image.sum()/recon_image.sum())
        print("Percent of Detector Counts in Section: ", forward_project.sum()/counts.sum())

        for ax, image, climit in zip([ax_p, ax_i], [forward_project, masked_image], [p_limits, i_limits]):
            if norm:
                img = ax.imshow(image, cmap='magma', origin='upper', interpolation='nearest',
                          vmin=climit[0], vmax=climit[1])
            else:
                img = ax.imshow(image, cmap='magma', origin='upper', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(img, fraction=0.046, pad=0.04, ax=ax)

    fig.tight_layout()
    if not batch:
        plt.show()
    # plt.close(fig)  # TODO: Does this work?


def main():
    # npix = (121, 31)  # uninterpolated
    # npix = (241, 61)

    # ================= Define Spaces =================
    # npix = (121 + 120, 31 + 30)  # fuller_FoV  # TODO: default
    # center = (0, -10)  # fuller 120 mm  # TODO: default

    # npix = (241, 61 * 2)  # 100mm Fuller
    # center = (0, -40)  # 100mm Fuller

    # center_env = (0, -110)
    # env_npix = (40, 23)

    # ~ Apr 28 FoV and Beamstop
    npix = (101 + 100, 31 + 30)  # fuller_FoV
    center = (0, -10)  # fuller 120 mm
    center_env = (200/2 + 1 + 200/2, -10)  # beamstop
    env_npix = (101, 31)
    # ================= Define Spaces =================

    # see_projection('/home/justin/repos/sysmat/design/2021-03-18-2312_SP0.h5', choose_pt=1060, npix=npix)
    data_file = '/home/justin/Desktop/images/zoom_fixed/thor10_07.npz'  # overnight
    data_0cm = '/home/justin/Desktop/images/recon/thick07/0cm.npz'
    data_6cm = '/home/justin/Desktop/images/recon/thick07/6cm.npz'
    data_6cm_filt = '/home/justin/Desktop/images/recon/thick07/6cm_filt.npz'  # Filtered on C12 peaks
    data_12cm = '/home/justin/Desktop/images/recon/thick07/12cm.npz'

    # sysmat_fname = '/home/justin/repos/python3316/processing/tst_interp.npy'  # 100mm_full but just interp (241, 61)
    # sysmat_fname = '/home/justin/repos/python3316/processing/100mm_full_processed_F1S7.npy'
    # sysmat_fname = '/home/justin/repos/python3316/processing/100mm_fuller_FoV_processed_F1S7.npy'  # (241, 61 *2)
    # sysmat_fname = '/home/justin/repos/sysmat/design/120mm_wide_FoV_processed_no_smooth.npy'
    # sysmat_fname = '/home/justin/repos/sysmat/design/system_responses/120mm_wide_FoV_processed_F0_5S7.npy'
    # TODO: Above was default

    # TODO: Apr 28
    # sysmat_fname = '/home/justin/repos/sysmat/design/Apr28_FoV_F0_7S7.npy'
    sysmat_fname = '/home/justin/repos/sysmat/design/Apr28_FoV_beamstop.npy'
    # see_projection(sysmat_fname, choose_pt=np.prod(npix)//2, npix=2 * np.array(npix) - 1)

    iterations = 30
    params = image_reconstruction_full(sysmat_fname, data_6cm_filt,
                                       npix,  # obj_pxls
                                       env_pxls=env_npix,  # tot  # TODO: Comment out to remove other
                                       obj_center=center,
                                       env_center=center_env,  # tot  # TODO: Comment out
                                       pxl_sze=(1, 2),  # TODO: usually (1, 10)
                                       filt_sigma=[0.5, 0.5],  # vertical, horizontal 0.25, 0.5
                                       # edge_correction=True,
                                       # sides_interp=True,
                                       nIterations=iterations)
    obj_params = params[0]
    plt.plot(obj_params[1], np.sum(obj_params[0], axis=0))
    plt.title("Projection Along Beam ({n} iterations)".format(n=iterations))
    plt.xlabel("Distance [mm]")
    plt.show()

    # previous was [50, 141, 50]
    # tot_recon = obj_params[0].sum() + params[1][0].sum()
    # print("Total recon counts FoV: ", obj_params[0].sum()/tot_recon)
    # print("Total recon counts BS: ", params[1][0].sum()/tot_recon)
    split_image_and_project(sysmat_fname, data_6cm_filt, obj_params[0], section_width_x=[40, 121, 40],
                            norm=True)
    # TODO: 3d image recon
    # TODO: sysmat_fname needs to be sliced for JUST the object


def batch_main():
    # ~ Apr 28 FoV and Beamstop
    npix = (101 + 100, 31 + 30)  # fuller_FoV
    center = (0, -10)  # fuller 120 mm
    center_env = (200 / 2 + 1 + 200 / 2, -10)  # beamstop
    env_npix = (101, 31)

    base_load_folder = '/home/justin/Desktop/processed_data/mm_runs/'
    base_save_folder = '/home/justin/Desktop/images/Apr19/new_fov_response_S2/mm_runs_recon/'  # fov, regions, sections

    sysmat_fname = '/home/justin/repos/sysmat/design/Apr28_FoV_beamstop.npy'

    # pos = np.arange(40, 61)
    pos = np.arange(40, 41)
    # slices = [None] * pos.size

    for id, p in enumerate(pos):
        if id == 0:
            batch = True
        else:
            batch = False
        data_file = base_load_folder + 'pos' + str(p) + 'mm_Apr27.npz'
        fov_name = base_save_folder + 'fov/pos' + str(p) + 'mm'
        iterations = 60
        # params, slice
        params = image_reconstruction_full(sysmat_fname, data_file,
                                           npix,  # obj_pxls
                                           env_pxls=env_npix,  # tot  # TODO: Comment out to remove other
                                           obj_center=center,
                                           env_center=center_env,  # tot  # TODO: Comment out
                                           pxl_sze=(1, 2),  # TODO: usually (1, 10)
                                           filt_sigma=[0.5, 0.5],  # vertical, horizontal 0.25, 0.5
                                           # batch_fname=fov_name,
                                           show_plot=batch,
                                           nIterations=iterations)
        # slices[id] = slice

    # x_vals = params[0][1]
    # print("x_vals size: ", x_vals.size)
    # print("Slice 0 size: ", slices[0].size)
    # for sli, p in zip(slices, pos):
    #     plt.plot(x_vals, sli, label='pos {p} mm'.format(p=p))
    # plt.xlabel('[mm]')
    # plt.ylabel('Counts')
    # plt.title("Projection Along Beam at 10 cm")
    # plt.legend(loc='best')
    # plt.show()


if __name__ == "__main__":
    # main()
    batch_main()


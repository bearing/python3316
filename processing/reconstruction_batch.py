import numpy as np
import time
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
# from scipy.stats import linregress, moment
import tables
import sys
# Date initialized: May 31


def compute_mlem_full(sysmat, counts, dims,
                      sensitivity=None,
                      det_correction=None,
                      initial_guess=None,
                      nIterations=10,
                      filter='gaussian',
                      filt_sigma=1,
                      verbose=True,
                      **kwargs):
    """Counts is a projection. Dims is a 2 tuple list of dimensions for each region, sensitivity normalizes iterations
    to detector sensitivity from system response, det_correction is a calibration of det pixel response, initial guess
    is initial guess for image (must be given as a list of image arrays like the output, nIterations is MLEM iterations,
    and filter/filter_sigma apply gaussian filter to ROI space (assumed to be first given region/dim"""

    tot_det_pixels, tot_img_pixels = sysmat.shape  # n_measurements, n_pixels

    tot_obj_plane_pxls = dims[0].prod()
    tot_reg_pxls = [tot_obj_plane_pxls]
    # tot_reg_pxls = []

    x_obj_pixels, y_obj_pixels = dims[0]

    if verbose:
        print("Total Measured Counts: ", counts.sum())
        print("Total Detector Pixels: ", tot_det_pixels)
        print("Total Image Pixels: ", tot_img_pixels)
        print("Check Level (Counts):", counts.sum() * 0.001 + 100)
        print("Standard Deviation (Counts):", np.std(counts))

    regions = [' Object ']
    reg_ids = np.arange(len(dims))
    for rid in reg_ids[1:]:
        regions.append(' Region ' + str(rid) + ' ')

    for region_str, region_dims in zip(regions[1:], dims[1:]):
        if verbose:
            print("Total", region_str, 'Pixels: ', np.prod(region_dims))
        tot_reg_pxls.append(np.prod(region_dims))

    if sensitivity is None:
        sensitivity = np.ones(tot_img_pixels)

    if det_correction is None:
        det_correction = np.ones(tot_det_pixels)

    sensitivity = sensitivity.ravel()
    det_correction = det_correction.ravel()

    measured = counts.ravel() * det_correction

    recon_img = np.ones(tot_img_pixels)

    if initial_guess is None:
        recon_img_previous = np.ones(recon_img.shape)
    else:
        assert len(initial_guess) == len(regions),\
            "Initial {i} Guess Regions for Required {n} regions".format(i=len(initial_guess), n=len(regions))
        recon_img_previous = np.concatenate([img.ravel() for img in initial_guess])

    diff = np.ones(recon_img.shape) * np.mean(counts)  # NOTE: Added scaling with mean

    itrs = 0
    t1 = time.time()

    sysmat[sysmat == 0] = 0.01 * np.min(sysmat[sysmat != 0])

    # while itrs < nIterations and (diff.sum() > (0.001 * counts.sum() + 100)):
    # TODO: uncomment above. Why does this not work now?
    while itrs < nIterations: #  and (diff.sum() > (0.001 * counts.sum() + 100)):
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

    if verbose:
        print("Total Iterations: ", itrs)

    inds = np.cumsum(tot_reg_pxls)[:-1]  # for split function
    recons = np.split(recon_img, inds)  # these are raveled

    # print("tot_reg_pxls: ", tot_reg_pxls)
    # print("inds: ", inds)
    # print("Length of recons: ", len(recons))
    # print("Print reg_ids: ", reg_ids)
    # print("dims: ", dims)

    for r in reg_ids:
        if verbose:
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


# ===========Batch Versions===========
class Reconstruction(object):
    def __init__(self, sysmat_filename, region_pxls, region_centers, pxl_sizes, plot_locations=None):
        n_regions, reg_dims = region_pxls.shape
        assert reg_dims == 2, "Expected (n, 2) shape for region_pxls. Got {s} instead.".format(s=region_pxls.shape)
        self.n_regions = n_regions
        if pxl_sizes.size == 1 and self.n_regions > 1:
            self.pxl_sizes = np.repeat(pxl_sizes, self.n_regions)
        else:
            self.pxl_sizes = pxl_sizes
        self.region_dims = region_pxls
        self.region_centers = region_centers

        # Needed for plot limits
        self.extent_x = self.region_centers[:, 0][:, np.newaxis] + \
                        (np.array([-1, 1]) * (self.region_dims[:, 0] * self.pxl_sizes)[:, np.newaxis]) / 2
        self.extent_y = self.region_centers[:, 1][:, np.newaxis] + \
                        (np.array([-1, 1]) * (self.region_dims[:, 1] * self.pxl_sizes)[:, np.newaxis]) / 2
        self.figure, self.axes, self.imgs, self.cbars = self.initialize_figures(plot_locations=plot_locations)
        self.line_projections = np.zeros([1, region_pxls[0, 1]])  # first must be object FoV

        self.sysmat = self.load_sysmat_from_file(sysmat_filename)
        self.counts = np.ones([48, 48])
        self.recons = [None] * self.n_regions

    def create_hdf5_file(self):  # TODO: For batch processing, save image recon in a stack, do not put in init
        pass

    def push_to_hdf5(self):  # TODO: push recons to hdf5 for batch processing, or maybe better to save individual anyway
        pass

    def initialize_figures(self, plot_locations=None):  # , line_project_regions=None):
        """plot_locations is the linearized indices (row order) of each region in self.region_dims for
        a 3x3 grid. Line_project_regions indicates which regions are line plots and not images"""
        # TODO: Allow for line projections
        x_labels = ['Beam [mm]']
        y_labels = ['Vertical [mm]']
        for i in np.arange(1, self.n_regions):
            x_labels.append('R' + str(i) + ' axis 0 [mm]')
            y_labels.append('R' + str(i) + ' axis 1 [mm]')

        # extent_x = self.region_centers[:, 0][:, np.newaxis] + \
        #           (np.array([-1, 1]) * (self.region_dims[:, 0] * self.pxl_sizes)[:, np.newaxis])/2
        # extent_y = self.region_centers[:, 1][:, np.newaxis] + \
        #            (np.array([-1, 1]) * (self.region_dims[:, 1] * self.pxl_sizes)[:, np.newaxis])/2

        fig = plt.figure(figsize=(18, 9), constrained_layout=False)
        cols = 3  # hardcoded for now
        rows = 3  # int(np.ceil(self.n_regions / cols))
        gs = fig.add_gridspec(nrows=rows, ncols=cols)

        axes_objs = []
        img_objs = []
        cbars = []

        if plot_locations is None:
            plot_locations = np.arange(self.n_regions)
        assert plot_locations.size == self.n_regions, "{p} plot locations does not fit {n} expected " \
                                                      "regions".format(p=plot_locations.size, n=self.n_regions)

        for rid, (r_dims, p_loc, x_label, y_label, rng_x, rng_y) in \
                enumerate(zip(self.region_dims, plot_locations, x_labels, y_labels, self.extent_x, self.extent_y)):

            # ax = fig.add_subplot(gs[row, col])
            ax = fig.add_subplot(gs[np.unravel_index(p_loc, (rows, cols))])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if id != 0:
                ax.set_title('Region ' + str(rid) + ' Image')
            img = ax.imshow(np.ones(r_dims[::-1]), cmap='magma', origin='upper',
                            interpolation='nearest', extent=np.append(rng_x, rng_y))

            cbars.append(fig.colorbar(img, fraction=0.046, pad=0.04, ax=ax))

            axes_objs.append(ax)
            img_objs.append(img)
        fig.tight_layout()
        return fig, axes_objs, img_objs, cbars

    def mlem_reconstruct(self, data_file, previous_iter=0, **kwargs):
        """data_file points to counts data file (projection), previous_iter allows for pausing the recon.
        **kwargs include (for compute_mlem_full) det_correction which experimentally corrects per pixel, initial_guess which must be
        list of n_regions in size (the output) of this function useful for saving between iterations, nIterations
         (10 default), verbose to print diagnostic info, filter (default gaussian) which filters between iterations,
         and then kwargs for gaussian_filter from scipy.ndimage.filter"""
        try:
            niter = kwargs['nIterations']
        except Exception as e:
            print(e)
            niter = 10

        try:
            kern = kwargs['filt_sigma']
        except Exception as e:
            print(e)
            kern = 1
        self.axes[0].set_title('Object FOV ({n} Iterations, kernel: {k})'.format(n=previous_iter + niter, k=kern))
        # First axis is always of the obj FoV (and not background)

        if not previous_iter:  # i.e. starting from scratch
            data = np.load(data_file)
            self.counts = data['image_list']

        self.recons = compute_mlem_full(self.sysmat, self.counts, self.region_dims,
                                        sensitivity=np.sum(self.sysmat, axis=0), **kwargs)
        return niter  # Useful for prev_iter i.e. pausing recon and saving at certain iterations

    def update_title(self, region_number, new_name):
        self.axes[region_number].set_title(new_name)

    def update_plots(self, norm_plot=False):  # show_plot=False):
        # TODO: set_data for line plot or image
        min_val, max_val = self.global_limits(self.recons)

        for region_image, recon, cbar in zip(self.imgs, self.recons, self.cbars):
            region_image.set_data(recon)

            if norm_plot:  # normalize to global min/max
                # cbar.set_clim(vmin=min, vmax=max)
                region_image.set_clim(vmin=min_val, vmax=max_val)
            else:  # normalize to RoI min/max
                # cbar.set_clim(vmin=recon.min(), vmax=recon.max())
                region_image.set_clim(vmin=recon.min(), vmax=recon.max())
            cbar.draw_all()
            # plt.draw()
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

    # @staticmethod
    def show_plots(self):  # TODO: Fix this, it doesn't work after first call
        # self.figure.show()
        self.figure.show()
        # plt.pause(1)

    def save_figure(self, fname):
        """fname is the desired saved file name. Automatically adds .png extension"""
        self.figure.savefig(fname + '.png')

    def save_image_data(self, fname):
        """fname is the desired saved file name. Saves only object recon"""
        np.save(fname, self.recons[0])

    def global_limits(self, recons):
        min_val = np.inf
        max_val = 0
        for region in recons:
            if region.min() < min_val:
                min_val = region.min()
            if region.max() > max_val:
                max_val = region.max()
        return min_val, max_val

    def load_sysmat_from_file(self, filename):
        sysmat = load_sysmat(filename)
        total_expected_pxls = np.product(self.region_dims.T, axis=0).sum()
        assert total_expected_pxls == sysmat.shape[1], \
            "Mismatch between expected pixels, {o}, and response: {r}".format(o=total_expected_pxls,
                                                                              r=sysmat.shape)
        return sysmat


def main():
    # system_response = '/home/justin/repos/sysmat/design/june1_full_response.npy'
    system_response = '/home/justin/repos/sysmat_current/sysmat/design/june15_full_response.npy'
    # pixels
    fov = [201, 61]
    top = [101, 39]
    bot = [101, 31]
    table = [59, 23]
    beamport = [101, 3]
    beamstop = [101, 31]

    # centers
    fc = [0, -10]
    tc = [0, 61]
    bc = [0, -71]
    tbc = [0, -110]
    bpc = [-601, -10]
    bsc = [201, -10]

    region_pixels = np.array([fov, top, bot, table, beamport, beamstop])
    region_centers = np.array([fc, tc, bc, tbc, bpc, bsc])
    pxl_szes = np.array([1, 2, 2, 10, 10, 2])
    plot_locations = np.array([4, 1, 7, 8, 3, 5])

    step_recon = Reconstruction(system_response, region_pixels, region_centers, pxl_szes, plot_locations=plot_locations)

    niters = 30
    filter = 'gaussian'
    filt_sigma = [0.5, 0.5]  # filt_sigma =[0.5, 0.5]
    verbose = True

    # def compute_mlem_full(sysmat, counts, dims, sensitivity=None, det_correction=None, initial_guess=None,
    # nIterations=10, filter='gaussian', filt_sigma=1, verbose=True, **kwargs):

    file_prefix = '/home/justin/Desktop/processed_data/mm_runs_june1/pos'
    file_suffix = 'mm_June1.npz'
    # data_file = '/home/justin/Desktop/processed_data/mm_runs_june1/pos50mm_June1.npz'

    save_prefix = '/home/justin/Desktop/processed_data/full_system_recons_june15/filt1/'
    # filt1/ or filt0_5, june 1 or 15
    save_suffix = 'mm'

    # steps = np.array([50, 60])
    steps = np.arange(39, 61+1)
    line_plot_data = np.zeros([steps.size, fov[0]])

    for sid, step in enumerate(steps):
        data_file = file_prefix + str(step) + file_suffix
        save = save_prefix + str(step) + save_suffix
        step_recon.mlem_reconstruct(data_file, nIterations=niters, filter=filter, filt_sigma=filt_sigma,
                                    verbose=verbose)
        step_recon.update_plots()
        # step_recon.save_image_data(save + str(step))  # TODO: Recomment
        # step_recon.save_figure(save + str(step))  # TODO: Recomment

        fov_image = step_recon.recons[0]
        line_plot_data[sid] = np.mean(fov_image[(35-2):(35+2), :], axis=0)
        # step_recon.show_plots()
    plt.show()

    # Line Projection Start
    x_proj_range = np.linspace(step_recon.extent_x[0][0], step_recon.extent_x[0][1], fov[0])
    for sid, step in enumerate(steps[::2]):
        current_line = line_plot_data[2 * sid]
        plt.plot(x_proj_range, current_line/np.max(current_line), label="{z} mm".format(z=step))

    plt.xlabel('[mm]')
    plt.ylabel('Counts')
    plt.legend(loc='best')
    plt.title("Sum Projection Along Beam Max (filter=0.5)")  # TODO: dynamically change title account filter

    plt.show()
    # Line Projection End

    # step_recon.mlem_reconstruct(data_file, nIterations=niters, filter=filter, filt_sigma=filt_sigma, verbose=verbose)
    # step_recon.update_plots()
    # step_recon.show_plots()

    save = '/home/justin/Desktop/processed_data/full_system_recons_june1/tst'
    # step_recon.save_image_data(save)
    # step_recon.save_figure(save)
    # step_recon.show_plots()
    # TODO: Generate gifs of projections, full field images, and line plots


def main2():
    # system_response = '/home/justin/repos/sysmat/design/june1_full_response.npy'
    system_response = '/home/justin/repos/sysmat_current/sysmat/design/june22_full_response.npy'
    # pixels
    fov = [201, 61]
    top = [101, 39]
    bot = [101, 31]
    beamstop = [101, 31]

    # centers
    fc = [0, -10]
    tc = [0, 61]
    bc = [0, -71]
    bsc = [201, -10]

    region_pixels = np.array([fov, top, bot, beamstop])
    region_centers = np.array([fc, tc, bc, bsc])

    pxl_szes = np.array([1, 2, 2, 2])
    plot_locations = np.array([4, 1, 7, 5])

    step_recon = Reconstruction(system_response, region_pixels, region_centers, pxl_szes, plot_locations=plot_locations)

    niters = 60
    filter = 'gaussian'
    filt_sigma = [1, 1]  # filt_sigma =[0.5, 0.5]
    verbose = True

    # def compute_mlem_full(sysmat, counts, dims, sensitivity=None, det_correction=None, initial_guess=None,
    # nIterations=10, filter='gaussian', filt_sigma=1, verbose=True, **kwargs):

    file_prefix = '/home/justin/Desktop/processed_data/mm_runs_june1/pos'
    file_suffix = 'mm_June1.npz'

    save_prefix = '/home/justin/Desktop/processed_data/full_system_recons_june15/filt1/'
    save_suffix = 'mm'

    # steps = np.array([50, 60])
    steps = np.arange(39, 61+1)
    line_plot_data = np.zeros([steps.size, fov[0]])

    for sid, step in enumerate(steps):
        data_file = file_prefix + str(step) + file_suffix
        step_recon.mlem_reconstruct(data_file, nIterations=niters, filter=filter, filt_sigma=filt_sigma,
                                    verbose=verbose)
        step_recon.update_plots()

        save = save_prefix + str(step) + save_suffix
        # step_recon.save_image_data(save + str(step))  # TODO: Recomment
        # step_recon.save_figure(save + str(step))  # TODO: Recomment

        fov_image = step_recon.recons[0]
        line_plot_data[sid] = np.mean(fov_image[(35-2):(35+2), :], axis=0)
    plt.show()

    # Line Projection Start
    x_proj_range = np.linspace(step_recon.extent_x[0][0], step_recon.extent_x[0][1], fov[0])
    for sid, step in enumerate(steps[::2]):
        current_line = line_plot_data[2 * sid]
        plt.plot(x_proj_range, current_line/np.max(current_line), label="{z} mm".format(z=step))

    plt.xlabel('[mm]')
    plt.ylabel('Counts')
    plt.legend(loc='best')
    plt.title("Sum Projection Along Beam Max (filter={f}))".format(f=str(filt_sigma[0])))
    # TODO: dynamically change title account filter

    plt.show()
    # Line Projection End


def main3(system_response, det_correction_fname=None, f_sig=(0.5, 0.5)):
    # system_response = '/home/justin/repos/sysmat_current/sysmat/design/june30_full_response.npy'
    # June30 was before det orientation fixed
    # system_response = '/home/justin/repos/sysmat_current/sysmat/design/july6_full_response.npy'
    # July 6: unfolded response, det orientation fixed
    # system_response = '/home/justin/repos/sysmat_current/sysmat/design/july6_full_response_folded.npy'
    # July 6: folded response, orientation fixed, gauss_scale = 4

    # pixels
    fov = [201, 61]
    top = [101, 31]  # [101, 31] for july 20. [101, 39] for july 6.
    bot = [101, 31]
    beamstop = [101, 31]

    # centers
    fc = [0, -10]
    tc = [0, 61]
    bc = [0, -71]
    bsc = [201, -10]

    region_pixels = np.array([fov, top, bot, beamstop])
    region_centers = np.array([fc, tc, bc, bsc])

    pxl_szes = np.array([1, 2, 2, 2])
    plot_locations = np.array([4, 1, 7, 5])

    step_recon = Reconstruction(system_response, region_pixels, region_centers, pxl_szes, plot_locations=plot_locations)

    det_correction = None  # TODO: Added:July 20
    if det_correction_fname is not None:
        det_correction = np.load(det_correction_fname)

    niters = 60
    filter = 'gaussian'
    filt_sigma = np.array(f_sig)
    # filt_sigma = [0.5, 0.5]  # filt_sigma =[0.5, 0.5]
    verbose = True

    # def compute_mlem_full(sysmat, counts, dims, sensitivity=None, det_correction=None, initial_guess=None,
    # nIterations=10, filter='gaussian', filt_sigma=1, verbose=True, **kwargs):

    file_prefix = '/home/justin/Desktop/processed_data/mm_runs_june1/pos'
    file_suffix = 'mm_June1.npz'

    save_prefix = '/home/justin/Desktop/processed_data/full_system_recons_june15/filt1/'
    save_suffix = 'mm'

    # steps = np.array([50, 60])
    steps = np.arange(39, 61+1)
    line_plot_data = np.zeros([steps.size, fov[0]])

    for sid, step in enumerate(steps):
        data_file = file_prefix + str(step) + file_suffix
        step_recon.mlem_reconstruct(data_file, nIterations=niters, filter=filter, filt_sigma=filt_sigma,
                                    verbose=verbose, det_correction=det_correction)  # TODO: July 20, added det_correct
        step_recon.update_plots()

        save = save_prefix + str(step) + save_suffix
        # step_recon.save_image_data(save + str(step))  # TODO: Recomment
        # step_recon.save_figure(save + str(step))  # TODO: Recomment

        fov_image = step_recon.recons[0]
        line_plot_data[sid] = np.mean(fov_image[(35-2):(35+2), :], axis=0)
    plt.show()

    # Line Projection Start
    x_proj_range = np.linspace(step_recon.extent_x[0][0], step_recon.extent_x[0][1], fov[0])
    for sid, step in enumerate(steps[::2]):
        current_line = line_plot_data[2 * sid]
        plt.plot(x_proj_range, current_line/np.max(current_line), label="{z} mm".format(z=step))

    plt.xlabel('[mm]')
    plt.ylabel('Counts')
    plt.legend(loc='best')
    plt.title("Sum Projection Along Beam Max (filter={f}))".format(f=str(filt_sigma[0])))
    # TODO: dynamically change title account filter

    plt.show()
    # Line Projection End


if __name__ == "__main__":
    # det_correction = '/home/justin/Desktop/july20/det_correction/det_correction_no_mid.npy'
    det_correction = '/home/justin/Desktop/july20/det_correction/det_correction_mid.npy'
    # det_correction = None

    # system_response = '/home/justin/repos/sysmat_current/sysmat/design/july6_full_response_folded.npy'
    # July 6: folded response, orientation fixed, gauss_scale = 4

    # system_response = '/home/justin/Desktop/july20/full_sysmats/july20_full_response_folded.npy'
    # July 20: folded response, orientation fixed, gauss_scale = 4, all other regions too

    # system_response = '/home/justin/Desktop/july20/full_sysmats/july20_full_response_eavg.npy'
    # July 20: unfolded response, orientation fixed, gauss_scale = 4, all other regions too, eavg
    # TODO: e_avg is FUCKED

    system_response = '/home/justin/Desktop/july20/full_sysmats/july20_full_response_pavg.npy'
    # July 20: unfolded response, orientation fixed, gauss_scale = 4, all other regions too, pavg
    # TODO: p_avg is FUCKED too

    f_sig = (2, 2)
    main3(system_response, det_correction_fname=det_correction, f_sig=f_sig)


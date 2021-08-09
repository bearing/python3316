import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from new_basis.sysmat_tools.det_correction import flip_det
# from scipy.stats import linregress, moment
import tables
# Date initialized: August 1


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

    # sysmat[sysmat == 0] = 0.01 * np.min(sysmat[sysmat != 0])  # Done ONCE in class

    # TODO: Maybe make it stop if difference between iterations changes by 1% or less of total FoV counts?
    while itrs < nIterations:  # and (diff.sum() > (0.001 * counts.sum() + 100)):
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
    beam_current = 0.1 * (6.24 * (10**18)) * (10**(-9))  # 0.1 nA
    acquisition_time = 60  # seconds
    # total_protons = beam_current * acquisition_time

    def __init__(self, sysmat_filename, region_pxls, region_centers, pxl_sizes, plot_locations=None,
                 n_protons=None):
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
        self.sysmat[self.sysmat == 0] = 0.01 * np.min(self.sysmat[self.sysmat != 0])

        self.fraction_acquisition_time = 1
        if n_protons is not None:
            self.total_protons = n_protons  # updates acquisition time, useful for sampling

        self.counts = np.ones([48, 48])  # real data. loaded from file
        self.simulated_counts = np.ones_like(self.counts)  # TODO: Make functions to simulate target number of counts
        self.recons = [np.zeros([1, 1])] * self.n_regions

    @property
    def fraction_acquisition_time(self):
        return self._fraction_acquisition_time

    @fraction_acquisition_time.setter
    def fraction_acquisition_time(self, value):
        self._fraction_acquisition_time = value

    @property
    def total_protons(self):
        return self.fraction_acquisition_time * self.beam_current * self.acquisition_time

    @total_protons.setter
    def total_protons(self, value):  # Set this to set fraction acquisition time
        # prev_acq_time = self.fraction_acquisition_time
        # scalar = value / self.total_protons  # scaling factor to new protons from old
        # self.acquisition_time = prev_acq_time * scalar  # update acquisition time
        self.fraction_acquisition_time = value / (1.0 * self.total_protons)

    def initialize_figures(self, plot_locations=None):  # , line_project_regions=None):
        """plot_locations is the linearized indices (row order) of each region in self.region_dims for
        a 3x3 grid. Line_project_regions indicates which regions are line plots and not images"""
        # TODO: Allow for line projections
        x_labels = ['Beam [mm]']
        y_labels = ['Vertical [mm]']
        for i in np.arange(1, self.n_regions):
            x_labels.append('R' + str(i) + ' axis 0 [mm]')
            y_labels.append('R' + str(i) + ' axis 1 [mm]')

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

    def _simulate_projections(self, measured_counts):  #
        pixel = np.repeat(np.arange(measured_counts.size) + 1, measured_counts.ravel().astype('int'))
        # Generate new sample for each bin based on mean from measured counts

        sampler = np.random.default_rng()
        simulations = sampler.uniform(size=pixel.size)
        # print("Pixel.size:", pixel.size)
        check_pass = (simulations < self.fraction_acquisition_time)  # is it in the window?

        resamp = np.bincount(pixel * check_pass, minlength=measured_counts.size + 1)[1:]
        # 0 bin is count is outside the time window

        return resamp.reshape(measured_counts.shape)

    def flip_det(self):
        pass

    def load_projections(self, data_files, *args, correction=True, simulate_n_protons=None, **kwargs):
        """Loads multiple energy bin projections. Data files must be a list or tuple of file paths.
        Correction applies correction, simulate_n_protons if not set to none simulates that many protons using proj
        file for mean sample rates (bootstrap)"""
        # flip_det(proj_array, ind, flip_ud=False, n_rot=1, ndets=(4, 4), det_pxls=(12, 12))
        self.counts.fill(0.0)

        if simulate_n_protons:
            self.total_protons = simulate_n_protons
            # print("Total Protons: ", self.total_protons)
            # print("Fractional Acquisition Time: ", self.fraction_acquisition_time)

        if isinstance(data_files, str):
            data_files = [data_files]

        for data_file in data_files:
            data = np.load(data_file)
            new_counts = data['image_list']
            if correction:
                try:
                    new_counts = flip_det(new_counts, *args, **kwargs)
                except Exception as e:
                    print("Corrected! ")
                    new_counts = flip_det(new_counts, 11, **kwargs)  # module SID 11 was plugged in incorrectly

            if simulate_n_protons:
                print("Simulating!")
                # new_counts = self._simulate_projections(new_counts, int(simulate_n_protons))
                new_counts = self._simulate_projections(new_counts)
            self.counts += new_counts
        print("Total projected counts: ", self.counts.sum())
        print("Fractional Acquisition Time: ", self.fraction_acquisition_time)
        print("Total Protons: ", self.total_protons)

    def mlem_reconstruct(self, previous_iter=0, **kwargs):
        """previous_iter allows for pausing the recon. **kwargs include (for compute_mlem_full) det_correction which
        experimentally corrects per pixel, initial_guess which must be list of n_regions in size (the output) of this
         function useful for saving between iterations, nIterations (10 default), verbose to print diagnostic info,
         filter (default gaussian) which filters between iterations, and then kwargs for gaussian_filter
         from scipy.ndimage.filter"""
        try:
            niter = kwargs['nIterations']
        except Exception as e:
            print(e)
            niter = 10
            print("Using {iter} max iterations instead.".format(iter=niter))

        try:
            kern = kwargs['filt_sigma']
        except Exception as e:
            print(e)
            kern = 1
        self.axes[0].set_title('Object FOV ({n} Iterations, kernel: {k})'.format(n=previous_iter + niter, k=kern))
        # First axis is always of the obj FoV (and not background)

        # if not previous_iter:  # i.e. starting from scratch
        #     data = np.load(data_file)
        #     self.counts = data['image_list']

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
        """Used to scale colorbars such that they are all identicla for all plots"""
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


def main(system_response, *args, regions=('r0', 'r1'), det_correction_fname=None,
         f_sig=(0.5, 0.5), save_steps=False, save_one_stack=False,
         **kwargs):
    """For use with aug responses"""
    region_fnames = {'r0': 'carbon_scatter/', 'r1': 'carbon/', 'r2': 'oxygen_scatter/', 'r3': 'oxygen/'}
    if isinstance(regions, str):
        regions = [regions]

    # pixels
    fov = [201, 61]
    top = [101, 39]  # [101, 31] for july 20. [101, 39] for july 6.
    bot = [101, 31]
    tbl = [41, 23]
    beamstop = [101, 31]
    beamport = [101, 31]

    # centers
    fc = [0, -10]
    tc = [0, 61]
    bc = [0, -71]
    tbc = [200, -110]  # (x, z)
    bsc = [201, -10]
    bpc = [-201, -10]

    region_pixels = np.array([fov, top, bot, tbl, beamstop, beamport])
    region_centers = np.array([fc, tc, bc, tbc, bsc, bpc])

    pxl_szes = np.array([1, 2, 2, 10, 2, 2])
    plot_locations = np.array([4, 1, 7, 8, 5, 3])

    step_recon = Reconstruction(system_response, region_pixels, region_centers, pxl_szes, plot_locations=plot_locations)

    det_correction = None
    if det_correction_fname is not None:
        det_correction = np.load(det_correction_fname)

    niters = 30  # normally 60
    filter = 'gaussian'
    filt_sigma = np.array(f_sig)
    # filt_sigma = [0.5, 0.5]
    verbose = True

    # def compute_mlem_full(sysmat, counts, dims, sensitivity=None, det_correction=None, initial_guess=None,
    # nIterations=10, filter='gaussian', filt_sigma=1, verbose=True, **kwargs):

    base_folder = '/home/justin/Desktop/final_projections/'
    file_prefix = 'pos'
    file_suffix = 'mm_Aug1.npz'

    # save_prefix = '/home/justin/Desktop/final_images/test/'  # TODO: Always check this
    save_prefix = '/home/justin/Desktop/final_images/full_5_107/'  # full, carbon, oxygen
    save_suffix = 'mm'

    # steps = np.array([65, 66])
    steps = np.arange(0, 101)
    line_plot_data = np.zeros([steps.size, fov[0]])

    img_stack = np.zeros(steps.size)
    if save_one_stack:
        img_stack = np.zeros([steps.size, fov[1], fov[0]])

    for sid, step in enumerate(steps):
        data_file = []
        for region in regions:
            print("Region: ", region_fnames[region])
            data_file.append(base_folder + region_fnames[region] + file_prefix + str(step) + file_suffix)
        step_recon.load_projections(data_file, *args, **kwargs)

        step_recon.mlem_reconstruct(nIterations=niters, filter=filter, filt_sigma=filt_sigma,
                                    verbose=verbose, det_correction=det_correction)
        step_recon.update_plots()

        save = save_prefix + 'pos' + str(step) + save_suffix
        if save_steps:
            step_recon.save_image_data(save)
            step_recon.save_figure(save)

        fov_image = step_recon.recons[0]

        if save_one_stack:
            img_stack[sid, ...] = fov_image

        line_plot_data[sid] = np.mean(fov_image[(35-2):(35+2), :], axis=0)
    plt.show()

    if save_one_stack:
        np.savez(save_prefix + 'stack',
                 images=img_stack,
                 steps=steps,
                 x_proj_range=np.linspace(step_recon.extent_x[0][0], step_recon.extent_x[0][1], fov[0]),
                 protons=step_recon.total_protons)

    # Line Projection Start
    x_proj_range = np.linspace(step_recon.extent_x[0][0], step_recon.extent_x[0][1], fov[0])
    for sid, step in enumerate(steps[::2]):
        current_line = line_plot_data[2 * sid]
        plt.plot(x_proj_range, current_line/np.max(current_line), label="{z} mm".format(z=step))

    plt.xlabel('[mm]')
    plt.ylabel('Counts')
    plt.legend(loc='best')
    plt.title("Sum Projection Along Beam Max (filter={f}))".format(f=str(filt_sigma[0])))

    plt.show()
    # Line Projection End


if __name__ == "__main__":
    # det_correction = '/home/justin/Desktop/july20/det_correction/det_correction_no_mid.npy'
    det_correction = '/home/justin/Desktop/july20/det_correction/det_correction_mid.npy'
    # det_correction = None

    f_sig = (2, 2)  # (2, 2) usually
    # rgns = ('r0', 'r1')  # Carbon
    # rgns = ('r3')
    # rgns = ('r2', 'r3')  # oxygen
    rgns = ('r0', 'r1', 'r2', 'r3')
    det_correct = True
    flip = True
    rot = 0

    # system_response = '/home/justin/repos/python3316/final/aug3_full_response.npy'
    # system_response = '/home/justin/repos/python3316/final/aug3_full_response_s2.npy'  # FoV subsampling = 2
    # August 3. Includes FoV, Top, Bot, Table, Beamstop, Beamport
    system_response = '/home/justin/repos/python3316/final/aug6_full_response_s1.npy'

    main(system_response, regions=rgns, det_correction_fname=det_correction, f_sig=f_sig,
         save_steps=False,  # Set True
         save_one_stack=False,  # Set True
         correction=det_correct,
         flip_ud=flip,
         n_rot=rot,
         simulate_n_protons=5 * (10**7))
    # TODO:
    #  4. Add 1% stopping criteria to mlem_reconstruct
    #  5. Run and save for all steps and for: oxygen, carbon, oxygen + carbon
    #  8. Write resample routine
    #  9. Repeat 5-7 for less protons
    #  10. Fold in physics response for oxygen and carbon. Repeat 5-7 and 9 but only for oxygen and carbon



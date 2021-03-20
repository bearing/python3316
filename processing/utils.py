import numpy as np
from scipy.ndimage import gaussian_filter
import time
import tables


def load_h5file(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def generate_detector_centers_and_norms(layout, det_width=50, focal_length=350,
                                        x_dir=np.array([1, 0, 0]),
                                        y_dir=np.array([0, 1, 0]),
                                        focal_dir=np.array([0, 0, 1])):
    """Detector Width and Focal Length in mm. Focal_dir is direction of focal points of detectors"""
    alpha = 2 * np.arctan((det_width/2) / focal_length)
    rows, cols = layout

    scalar_cols = np.arange(-cols / 2 + 0.5, cols / 2 + 0.5)
    scalar_rows = np.arange(-rows / 2 + 0.5, rows / 2 + 0.5)

    x_sc = focal_length * np.sin(np.abs(scalar_rows) * alpha) * np.sign(scalar_rows)  # horizontal
    y_sc = focal_length * np.sin(np.abs(scalar_cols) * alpha) * np.sign(scalar_cols)  # vertical

    # print("x_sc: ", x_sc)
    # print("y_sc: ", y_sc)

    focal_pt = focal_length * focal_dir

    x_vec = np.outer(x_sc, x_dir)  # Start left (near beam port) of beam axis
    y_vec = np.outer(y_sc[::-1], y_dir)  # Start top row relative to ground

    # print("x_vec: ", x_vec)
    # print("y_vec: ", y_vec)

    centers = (y_vec[:, np.newaxis] + x_vec[np.newaxis, :]).reshape(-1, 3)

    # print("Centers: ", centers)

    centers[:, 2] = np.sqrt((focal_length**2) - np.sum(centers[:, :2] ** 2, axis=1)) * (-np.sign(focal_pt[2]))
    # TODO: This is not generic. Fix someday? Probably rotate afterward

    # print("Centers: ", centers + focal_pt)

    directions = norm_vectors_array(-centers, axis=1)
    shifted_centers = centers + focal_pt  # this is now relative to center
    return shifted_centers, directions


def interpolate_system_response(sysmat, x_img_pixels, save_fname='interp'):
    # n_pixels, n_measurements i.e. (1875, 2304)
    # tot_det_pixels, tot_img_pixels = sysmat.shape  # n_measurements, n_pixels
    tot_img_pixels, tot_det_pixels = sysmat.shape  # n_pixels, n_measurements
    y_img_pixels = tot_img_pixels // x_img_pixels

    x_interp_img_pixels = (2 * x_img_pixels-1)
    y_interp_img_pixels = (2 * y_img_pixels-1)
    interp_sysmat = np.zeros([x_interp_img_pixels * y_interp_img_pixels, tot_det_pixels], dtype=sysmat.dtype)

    for row in np.arange(y_img_pixels):  # start from top row, fill in known values and interp in-between x vals
        interp_rid = 2 * row * x_interp_img_pixels  # start
        orig_rid = row * x_img_pixels
        interp_sysmat[interp_rid:interp_rid + x_interp_img_pixels:2, :] = sysmat[orig_rid:orig_rid+x_img_pixels, :]

        interp_sysmat[(interp_rid+1):interp_rid + x_interp_img_pixels:2, :] = \
            (sysmat[orig_rid:(orig_rid + x_img_pixels-1), :] + sysmat[(orig_rid+1):orig_rid + x_img_pixels, :]) * 0.5
    # This can probably be combined with the above
    for row in np.arange(1, y_interp_img_pixels, 2):  # interp y img vals between known values
        interp_rid = row * x_interp_img_pixels
        a_rid = (row-1) * x_interp_img_pixels  # This is skipped by iteration (above rid)
        b_rid = (row+1) * x_interp_img_pixels  # (b)elow rid
        interp_sysmat[interp_rid:interp_rid+x_interp_img_pixels:2, :] = \
            (interp_sysmat[a_rid:a_rid+x_interp_img_pixels:2, :] + interp_sysmat[b_rid:b_rid+x_interp_img_pixels:2, :])\
            * 0.5

        interp_sysmat[(interp_rid + 1):interp_rid + x_interp_img_pixels:2, :] = \
            (interp_sysmat[a_rid:a_rid+x_interp_img_pixels-2:2, :] +
             interp_sysmat[(a_rid+1):a_rid + x_interp_img_pixels:2, :] +
             interp_sysmat[b_rid:b_rid + x_interp_img_pixels - 2:2, :] +
             interp_sysmat[(b_rid + 1):b_rid + x_interp_img_pixels:2, :]) * 0.25

    print("Interpolated Shape: ", interp_sysmat.shape)
    print("Nonzero values (percent): ", 1.0 * np.count_nonzero(interp_sysmat)/interp_sysmat.size)
    np.save(save_fname, interp_sysmat)


def generate_flat_detector_pts(layout, center, mod_spacing_dist):
    rows, cols = layout

    scalar_cols = np.arange(-cols / 2 + 0.5, cols / 2 + 0.5) * mod_spacing_dist
    scalar_rows = np.arange(-rows / 2 + 0.5, rows / 2 + 0.5) * mod_spacing_dist

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])

    # distance_mod_plane = system.collimator.colp + np.array([0, 0, -130]) + (25.4 * x)  # shift of center

    x_vec = np.outer(scalar_cols, x)  # Start left (near beam port) of beam axis
    y_vec = np.outer(scalar_rows[::-1], y)  # Start top row relative to ground

    return (y_vec[:, np.newaxis] + x_vec[np.newaxis, :]).reshape(-1, 3) + center


def norm_vectors_array(mat, axis=1):  # Must be rows of vectors
    return mat/np.sqrt(np.sum(mat**2, axis=axis, keepdims=True))


def compute_mlem(sysmat, counts, x_img_pixels, x_det_pixels=48, sensitivity=None,
                 det_correction=None,
                 nIterations=10,
                 filter='gaussian',
                 filt_sigma=1,
                 **kwargs):

    # print("Sysmat shape: ", sysmat.shape)
    print("Total Measured Counts: ", counts.sum())  # TODO: Normalize?

    tot_det_pixels, tot_img_pixels = sysmat.shape  # n_measurements, n_pixels
    y_img_pixels = tot_img_pixels//x_img_pixels
    y_det_pixels = tot_det_pixels//x_det_pixels

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
    recon_img_previous = np.zeros_like(recon_img)
    diff = 10**6 * np.ones_like(recon_img)
    outSum = np.zeros_like(recon_img)

    if sensitivity is None:
        sensitivity = np.ones(tot_img_pixels)

    itrs = 0
    t1 = time.time()

    while itrs < nIterations:  # and (diff.sum() > 0.001 * counts.sum() + 100):
        sumKlamb = sysmat.dot(recon_img)
        outSum = (sysmat * measured[:, np.newaxis]).T.dot(1/sumKlamb)
        recon_img *= outSum / sensitivity

        if itrs > 5 and filter == 'gaussian':
            recon_img = gaussian_filter(recon_img.reshape([y_img_pixels, x_img_pixels]), filt_sigma, **kwargs).ravel()
            # gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

        print('Iteration %d, time: %f sec' % (itrs, time.time() - t1))
        diff = np.abs(recon_img - recon_img_previous)
        print('Diff Sum: ', diff.sum())
        recon_img_previous = recon_img
        itrs += 1
    print("Total Iterations: ", itrs)
    return recon_img


def generate_PSFs(sysmat, buffer_xy, x_img_pixels, save_fname='psfs', **kwargs):
    tot_det_pixels, tot_img_pixels = sysmat.shape
    y_img_pixels = tot_img_pixels // x_img_pixels

    width = buffer_xy[0] * 2  # X
    height = buffer_xy[1] * 2  # Y
    pass  # TODO: Model after gaussian_smooth_response but varying gaussian


def make_gaussian(size, fwhm=1):  # f, center=None):
    """ Make a centered normalized square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    size = (np.ceil(size)//2 * 2) + 1  # rounds size to nearest odd integer
    x = np.arange(0, size, 1, float)  # size should really be an odd integer
    y = x[:, np.newaxis]

    x0 = y0 = (x[-1] + x[0])/2

    # fwhm = (4 * np.log(2)/6) * (vox**2)  # where vox is the length of a box in pixels
    gaussian = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    return gaussian/gaussian.sum()


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""
    # Our 2-dimensional distribution will be over variables X and Y
    # N = 40
    # X = np.linspace(-2, 2, N)
    # Y = np.linspace(-2, 2, N)
    # X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    # mu = np.array([0., 0.])
    # Sigma = np.array([[1., -0.5], [-0.5, 1.]])

    # Pack X and Y into a single 3-dimensional array
    # pos = np.empty(X.shape + (2,))
    # pos[:, :, 0] = X
    # pos[:, :, 1] = Y

    # The distribution on the variables X, Y packed into pos.
    # Z = multivariate_gaussian(pos, mu, Sigma)

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N


def gaussian_smooth_response(sysmat, x_img_pixels, *args, **kwargs):
    # assumption is that sysmat shape is (n_pixels, n_measurements) i.e. (1875, 2304)
    tot_img_pixels, tot_det_pixels = sysmat.shape  # n_pixels, n_measurements

    view = sysmat.T.reshape([tot_det_pixels, tot_img_pixels // x_img_pixels,  x_img_pixels])
    # TODO: Might not need to transpose in this way
    smoothed_reponse = np.copy(view)
    print("View shape: ", view.shape)

    kern = make_gaussian(*args, **kwargs)  # size, fwhm=1
    ksize = kern.shape[0]
    print("Kern: ", kern)
    buffer = int(np.floor(ksize/2))  # kernel is square for now

    # resmat * wgts[None,...]  where resmat is the (det_pxl, size, size) block
    for row in np.arange(buffer, (tot_img_pixels // x_img_pixels)-buffer):
        if row % 10 == 0:
            print("Row: ", row)
        upper_edge = row-buffer  # of region to multiply with kernel
        for col in np.arange(buffer, x_img_pixels-buffer):
            left_edge = col-buffer
            smoothed_reponse[:, row, col] = (view[:, upper_edge:upper_edge+ksize, left_edge:left_edge+ksize] *
                                             kern[None, ...]).sum(axis=(1, 2))
    # a1.swapaxes(0,2).swapaxes(0,1).reshape(m2.shape)
    return smoothed_reponse.transpose((1, 2, 0)).reshape(sysmat.shape)


def test_orientation():
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)

    centers, dirs = generate_detector_centers_and_norms(np.array([4, 4]), focal_length=350)
    for det_idx, det_center in enumerate(centers):
        print("Set det_center: ", det_center)
        print("Direction: ", dirs[det_idx])
        pass

    # ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2])
    ax.quiver(centers[:, 0], centers[:, 1], centers[:, 2], dirs[:, 0], dirs[:, 1], dirs[:, 2], length=20)
    ax.set_zlim(0, 130)
    plt.show()


def smooth_point_response(sysmat_filename, x_img_pixels, *args, h5file=True, **kwargs):
    if h5file:
        sysmat_file = load_h5file(sysmat_filename)
        sysmat = sysmat_file.root.sysmat[:]
    else:
        sysmat = np.load(sysmat_filename)

    size = args[0]
    try:
        fwhm = int(kwargs['fwhm']/2.355)
    except:
        fwhm = 1

    print("Sysmat Shape:", sysmat.shape)
    save_name = sysmat_filename[:sysmat_filename.find("SP")+3] + "_F" + str(fwhm) + "S" + str(size)
    # gaussian_smooth_response(sysmat, x_img_pixels, *args, **kwargs)
    np.save(save_name, gaussian_smooth_response(sysmat, x_img_pixels, *args, **kwargs))  # size, fwhm of kernel
    if h5file:
        sysmat_file.close()


def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))


def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]


if __name__ == "__main__":
    test_orientation()
    # smooth_point_response("/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0_interp.npy", 149, 7,
    #                      h5file=False, fwhm=2.355 * 1)  # 2.355 * spread defined in gaussian function (uncertainty)


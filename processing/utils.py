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

    focal_pt = focal_length * focal_dir

    x_vec = np.outer(x_sc, x_dir)  # Start left (near beam port) of beam axis
    y_vec = np.outer(y_sc[::-1], y_dir)  # Start top row relative to ground

    centers = (y_vec[:, np.newaxis] + x_vec[np.newaxis, :]).reshape(-1, 3)

    centers[:, 2] = np.sqrt((focal_length**2) - np.sum(centers[:, :2] ** 2, axis=1)) * (-np.sign(focal_pt[2]))
    # TODO: This is not generic. Fix someday? Probably rotate afterward

    directions = norm_vectors_array(-centers, axis=1)
    shifted_centers = centers + focal_pt  # this is now relative to center
    return shifted_centers, directions


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


def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))


def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]


def edge_gain(img, sides_interp=True):
    """Gain correction for a single module"""
    tmp = img.reshape(img.shape[0] // 12, 12, img.shape[1] // 12, 12).swapaxes(1, 2).reshape(-1, 12, 12)
    img_list = [tmp[ind] for ind in np.arange(16)]
    print("Image List length: ", img_list)

    for mod_pixels in img_list:
        y, x = mod_pixels.shape
        fh, sh = np.split(np.arange(1, x-1), 2)
        # Top Row first
        ul = (mod_pixels[1, fh].sum() / mod_pixels[0, fh].sum())  # upper left
        ur = (mod_pixels[1, sh].sum() / mod_pixels[0, sh].sum())  # upper right
        mod_pixels[0, fh] *= ul
        mod_pixels[0, sh] *= ur
        # Bottom Row
        ll = (mod_pixels[-2, fh].sum() / mod_pixels[-1, fh].sum())  # lower left
        lr = (mod_pixels[-2, sh].sum() / mod_pixels[-1, sh].sum())  # lower right
        mod_pixels[-1, fh] *= ll
        mod_pixels[-1, sh] *= lr
        if sides_interp:  # columns
            # th = fh  # top half
            # bh = sh  # bottom half
            ul2 = (mod_pixels[fh, 1].sum() / mod_pixels[fh, 0].sum())  # upper left
            ur2 = (mod_pixels[sh, 1].sum() / mod_pixels[sh, 0].sum())  # upper right
            ll2 = (mod_pixels[fh, -2].sum() / mod_pixels[fh, -1].sum())  # lower left
            lr2 = (mod_pixels[sh, -2].sum() / mod_pixels[sh, -1].sum())  # lower right
            mod_pixels[fh, 0] *= ul2  # upper left
            mod_pixels[sh, 0] *= ur2  # upper right
            mod_pixels[fh, -1] *= ll2  # lower left
            mod_pixels[sh, -1] *= lr2  # lower right
            ul = np.mean([ul, ul2])
            ur = np.mean([ur, ur2])
            ll = np.mean([ll, ll2])
            lr = np.mean([lr, lr2])
        else:
            mod_pixels[fh, 0] *= ul  # upper left
            mod_pixels[sh, 0] *= ur  # upper right
            mod_pixels[fh, -1] *= ll  # lower left
            mod_pixels[sh, -1] *= lr  # lower right
        mod_pixels[0, 0] *= ul
        mod_pixels[0, -1] *= ur
        mod_pixels[-1, 0] *= ll
        mod_pixels[-1, -1] *= lr

    return np.block([img_list[col:col + 4] for col in np.arange(0, len(img_list), 4)])


def append_responses(files, save_name='appended'):  # sysmat files
    tmp_list = list(range(len(files)))
    for fid, file in enumerate(files):
        if tables.is_hdf5_file(file):
            sysmat_file = load_h5file(file)
            tmp_list[fid] = sysmat_file.root.sysmat[:]
            sysmat_file.close()
        else:
            tmp_list[fid] = np.load(file)

        print("File {f} shape: {s}".format(f=fid, s=tmp_list[fid].shape))
    np.save(save_name, np.vstack(tmp_list))
    print("Final shape: ", np.vstack(tmp_list).shape)


if __name__ == "__main__":
    # test_orientation()
    files = ['/home/justin/repos/sysmat/design/2021-03-30-2347_SP0_F1S7.npy',
             '/home/justin/repos/sysmat/design/2021-03-23-2309_SP0_F1S7.npy',
             '/home/justin/repos/sysmat/design/2021-03-24-1651_SP0_F1S7.npy',
             '/home/justin/repos/sysmat/design/2021-03-30-2207_SP0.h5']  # Tab;e
    save_fname = '3_31_2021_tot'
    append_responses(files, save_name=save_fname)

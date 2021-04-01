import numpy as np
import time
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
# from scipy.stats import linregress, moment
import tables


def andy_mlem(sysMat, counts, correction=None, nIter=10, sens_j=None, ):
    """ this function computes iterations of MLEM
        it returns the image after nIter iterations

        sysMat is the system matrix, it should have shape:
            (n_measurements, n_pixels)
            it can be either a 2D numpy array, numpy matrix, or scipy sparse
            matrix
        counts is an array of shape (n_measurements) that contains the number
            of observed counts per detector bin

        sens_j is the sensitivity for each image pixel
            is this is None, uniform sensitivity is assumed
    """

    nPix = sysMat.shape[1]

    if sens_j is None:
        sens_j = np.ones(nPix)

    if correction is None:
        correction = np.ones(nPix)
    else:
        correction = correction.reshape([48 * 48])

    lamb = np.ones(nPix)
    lamb_previous = np.zeros(nPix)
    diff = 10 ** 6 * np.ones(nPix)
    outSum = np.zeros(nPix)
    iIter = 0
    print("npix: ", nPix)

    if counts is None:
        counts = np.ones(sysMat.shape[0])
    else:
        counts = np.fliplr(counts).reshape(48 * 48)

    backProj = (sysMat.dot(counts))
    print("Backproject shape: ", backProj.shape)

    # print
    # 'Computing Iterations'
    t1 = time.time()
    # for iIter in range(nIter):
    while diff.sum() > 0.001 * counts.sum() + 100 and iIter < nIter:
        sumKlamb = lamb.dot(sysMat.T)
        print("sumKlamb shape: ", sumKlamb.shape)
        # preF = sysMat.T * counts[:, np.newaxis]
        # print("preF: ", preF.shape)
        outSum = (counts/sumKlamb) * sysMat
        # outSum = (sysMat.T * counts[:, np.newaxis]).dot(1 / sumKlamb)
        lamb = lamb * outSum / sens_j
        lamb = lamb * correction

        if iIter > 5:
            # TODO: actual ndim = [150,50]
            lamb = lamb.reshape(150, 50)  # TODO: Fix this. Automate

            lamb = ndimage.gaussian_filter(lamb, 1)
            # lamb = ndimage.median_filter(lamb,3)
            # lamb = lamb.reshape(51*51)
            lamb = lamb.reshape(150 * 50)

        # print
        # 'Iteration %d, time: %f sec' % (iIter, time.time() - t1)
        # diff = abs(lamb - lamb_previous)
        # diff.sum()
        diff = np.sum(np.abs(lamb - lamb_previous))
        lamb_previous = lamb
        iIter += 1

    print("Iterations before stop: ", nIter)
    return lamb


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
    print("Total Image Pixels: ", tot_img_pixels)
    print("Total Detector Pixels: ", tot_det_pixels)
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
    return recon_img.reshape([y_img_pixels, x_img_pixels])


def compute_mlem2(sysmat, counts, obj_dims, env_dims, x_det_pixels=48, sensitivity=None,
                 det_correction=None,
                 nIterations=10,
                 filter='gaussian',
                 filt_sigma=1,
                 **kwargs):
    """Major difference is that this will also reconstruct response for environment. img and env dims in (x, y))"""
    # print("Sysmat shape: ", sysmat.shape)
    print("Total Measured Counts: ", counts.sum())  # TODO: Normalize?

    tot_det_pixels, tot_img_pixels = sysmat.shape  # n_measurements, n_pixels
    x_obj_pixels, y_obj_pixels = obj_dims
    x_env_pixels, y_env_pixels = env_dims

    tot_obj_plane_pxls = obj_dims.prod()
    tot_env_plans_pxls = env_dims.prod()

    y_det_pixels = tot_det_pixels//x_det_pixels

    print("Total Image Pixels: ", tot_img_pixels)
    print("Total Object Plane Pixels: ", tot_obj_plane_pxls)
    print("Total Environment Pixels: ", tot_env_plans_pxls)
    print("Total Detector Pixels: ", tot_det_pixels)
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
            recon_img[:tot_obj_plane_pxls] = \
                gaussian_filter(recon_img[:tot_obj_plane_pxls].reshape([y_obj_pixels, x_obj_pixels]),
                                        filt_sigma, **kwargs).ravel()
            # gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

        print('Iteration %d, time: %f sec' % (itrs, time.time() - t1))
        diff = np.abs(recon_img - recon_img_previous)
        print('Diff Sum: ', diff.sum())
        recon_img_previous = recon_img
        itrs += 1
    print("Total Iterations: ", itrs)
    return recon_img[:tot_obj_plane_pxls].reshape([y_obj_pixels, x_obj_pixels]), \
           recon_img[tot_obj_plane_pxls:].reshape([y_env_pixels, x_env_pixels])  # obj plane, environment


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


def main():
    correct = True

    sys_mat_fname = '/home/proton/repos/python3316/processing/2020-10-27-1019_SP4.h5'
    sys_mat_file = load_h5file(sys_mat_fname)
    sys_mat = sys_mat_file.root.sysmat[:]
    # sys_mat = gaussian_filter(sys_mat.T, sigma=1)  # John does this? Why?

    if correct:
        correction = np.load('th_uncalib_Oct31_flood.npy')
    else:
        correction = None

    flood = correction.mean()/correction  # correction
    raw_projection = np.load('step_run_5t6cm_Nov3.npy')
    # projection = raw_projection.mean()/raw_projection

    print("sys_mat shape: ", sys_mat.shape)
    print("correction shape: ", flood.shape)
    print("projection shape: ", raw_projection.shape)

    corrected = flood * raw_projection
    # corrected[corrected > (3 * corrected.mean())] = corrected.mean() * 3

    plt.imshow(raw_projection.T, cmap='jet', origin='upper', interpolation='nearest', aspect='equal')
    plt.colorbar()
    # plt.title('5-6 cm Projection Raw')
    # plt.show()

    andy_mlem(sys_mat, raw_projection, correction=flood, nIter=10)
    # plot_2D(result)
    sys_mat_file.close()


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


def image_reconstruction(img_pxls, data_file, center=(0, 0), pxl_sze=1, **kwargs):
    # filename = '/home/justin/repos/python3316/processing/2021-02-28-2345_SP0_F1S7.npy'  # original centered
    # filename = '/home/justin/repos/sysmat/design/2021-03-17-1523_SP0.h5'  # shifted +y by 25 mm
    filename = '/home/justin/repos/sysmat/design/2021-03-18-2312_SP0.h5'  # shifted -y by 25 mm
    # filename = '/home/justin/repos/sysmat/design/2021-03-23-2309_SP0_interp.npy'  # center = (0, -25)
    # filename = '/home/justin/repos/python3316/processing/2021_obj_appended.npy'

    if tables.is_hdf5_file(filename):
        sysmat_file_obj = load_h5file(filename)
        sysmat = sysmat_file_obj.root.sysmat[:].T
    else:
        sysmat = np.load(filename).T
    data = np.load(data_file)
    counts = data['image_list']

    img_pxl_x, img_pxl_y = img_pxls  # Careful about x and y with how python indexes

    recon = \
        compute_mlem(sysmat, counts, img_pxls[0], sensitivity=np.sum(sysmat, axis=0), **kwargs)
    # recon = \
    #    compute_mlem(sysmat, counts, img_pxls[0], sensitivity=np.sum(sysmat, axis=0), **kwargs)
    # recon[0] = 0
    # recon[-1] = 0
    extent_x = center[0] + (np.array([-1, 1]) * img_pxl_x / 2 * pxl_sze)
    extent_y = center[1] + (np.array([-1, 1]) * img_pxl_y / 2 * pxl_sze)

    fig = plt.figure(figsize=(12, 9))
    img = plt.imshow(recon, cmap='magma', origin='upper', interpolation='nearest',
               extent=np.append(extent_x, extent_y))
    ax = plt.gca()
    fig.colorbar(img, fraction=0.046, pad=0.04, ax=ax)
    plt.show()
    return recon, np.arange(extent_x[0], extent_x[1], pxl_sze) + 0.5 * pxl_sze, \
           np.arange(extent_y[0], extent_y[1], pxl_sze) + 0.5 * pxl_sze


def image_reconstruction2(sysmat_file, data_file, obj_pxls, env_pxls, obj_center=(0, 0), env_center=(0, - 130),
                          pxl_sze=(1, 10), **kwargs):
    """For use with compute_mlem2. Generates two images. Object plane and environment"""
    from matplotlib.gridspec import GridSpec

    # filename = '/home/justin/repos/sysmat/design/2021-03-18-2312_SP0.h5'  # shifted -y by 25 mm
    # data_file = '/home/justin/Desktop/images/zoom_fixed/thor10_07.npz'

    filename = sysmat_file

    if tables.is_hdf5_file(filename):
        sysmat_file_obj = load_h5file(filename)
        sysmat = sysmat_file_obj.root.sysmat[:].T
    else:
        sysmat = np.load(filename).T
    data = np.load(data_file)
    counts = data['image_list']

    obj_pxls = np.array(obj_pxls)
    env_pxls = np.array(env_pxls)

    print("Obj_pxls: ", obj_pxls)
    obj_pxl_x, obj_pxl_y = obj_pxls  # Careful about x and y with how python indexes
    env_pxl_x, env_pxl_y = env_pxls

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])  # object
    ax2 = fig.add_subplot(gs[0, 1])  # environment

    obj_recon, env_recon = compute_mlem2(sysmat, counts, obj_pxls, env_pxls,
                                         sensitivity=np.sum(sysmat, axis=0), **kwargs)

    obj_extent_x = obj_center[0] + (np.array([-1, 1]) * obj_pxl_x / 2 * pxl_sze[0])
    obj_extent_y = obj_center[1] + (np.array([-1, 1]) * obj_pxl_y / 2 * pxl_sze[0])
    obj_im = ax1.imshow(obj_recon, cmap='magma', origin='upper', interpolation='nearest',
                        extent=np.append(obj_extent_x, obj_extent_y))
    # ax1.axis('off')
    ax1.set_title('Object FOV')
    ax1.set_xlabel('Beam [mm]')
    ax1.set_ylabel('Vertical [mm]')
    fig.colorbar(obj_im, fraction=0.046, pad=0.04, ax=ax1)

    env_extent_x = env_center[0] + (np.array([-1, 1]) * env_pxl_x / 2 * pxl_sze[1])
    env_extent_y = env_center[1] + (np.array([-1, 1]) * env_pxl_y / 2 * pxl_sze[1])
    env_im = ax2.imshow(env_recon, cmap='magma', origin='upper', interpolation='nearest',
                        extent=np.append(env_extent_x, env_extent_y))
    # ax2.axis('off')
    ax2.set_title('Table')
    ax2.set_xlabel('Beam [mm]')
    ax2.set_ylabel('System Axis [mm]')
    fig.colorbar(env_im, fraction=0.046, pad=0.04, ax=ax2)
    fig.tight_layout()

    obj_params = [obj_recon, np.arange(obj_extent_x[0], obj_extent_x[1], pxl_sze[0]) + 0.5 * pxl_sze[0],
                  np.arange(obj_extent_y[0], obj_extent_y[1], pxl_sze[0]) + 0.5 * pxl_sze[0]]
    env_params = [env_recon, np.arange(env_extent_x[0], env_extent_x[1], pxl_sze[1]) + 0.5 * pxl_sze[1],
                  np.arange(env_extent_y[0], env_extent_y[1], pxl_sze[1]) + 0.5 * pxl_sze[1]]
    plt.show()
    return obj_params, env_params


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

    # ====Image Recon 1====
    # img, x, y = \
    #     image_reconstruction(npix, data_6cm, filt_sigma=[1, 1], center=center, pxl_sze=1, nIterations=100)
    #

    # plt.plot(x, np.sum(img, axis=0))
    # plt.title("Projection Along Beam")
    # plt.xlabel('[mm]')
    # plt.show()

    # plt.plot(y, np.sum(img, axis=1))
    # plt.title("Projection Along Orthogonal Axis")
    # plt.xlabel('[mm]')
    # plt.show()

    # ====Image Recon 2 ====
    center_env = (0, -110)  # (x, z)
    sysmat_fname = '/home/justin/repos/python3316/processing/3_31_2021_tot.npy'
    obj_params, env_params = image_reconstruction2(sysmat_fname, data_6cm,
                                                   obj_pxls=npix,
                                                   env_pxls=(21, 23),
                                                   obj_center=center,
                                                   env_center=center_env,
                                                   filt_sigma=[0.25, 1],
                                                   nIterations=30)
    # TODO: Modify this so it dynammically plots images of each region based on given system responses and envir.
    # TODO: Put flags so it automatically interpolates and smooths raw responses
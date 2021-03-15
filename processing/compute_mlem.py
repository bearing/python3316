import numpy as np
import time
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
# from scipy.stats import linregress, moment
import tables


def computeMLEM(sysMat, counts, correction=None, nIter=10, sens_j=None, ):
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

    computeMLEM(sys_mat, raw_projection, correction=flood, nIter=10)
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
    img = plt.imshow(sysmat[choose_pt].reshape([48, 48]), cmap='jet', origin='upper', interpolation='nearest', aspect='equal')
    plt.title("Projection", fontsize=14)
    # plt.xlabel('mm', fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.ylabel('mm', fontsize=14)
    # plt.yticks(fontsize=14)

    plt.colorbar(img, fraction=0.046 * (sysmat.shape[0]/sysmat.shape[1]), pad=0.04)
    plt.show()
    sysmat_file.close()


if __name__ == "__main__":
    # main()
    # sensivity_map('/home/proton/repos/python3316/processing/system_responses/2021-02-17-0252_SP4.h5', npix=[201, 51])
    # sensivity_map('/home/proton/repos/python3316/processing/system_responses/2021-02-23-1514_SP0.h5', npix=[75, 25])

    npix = np.array([75, 25])
    see_projection('/home/proton/repos/python3316/processing/system_responses/2021-02-23-1514_SP0.h5',
                   choose_pt=950, npix=npix)

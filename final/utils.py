import tables
import numpy as np
# import time


# ===============Interpolation ================
def system_matrix_interpolate(sysmat_filename, x_dim=75):
    """Kargs: x_img_pixels, save_fname, """
    sysmat_file = load_h5file(sysmat_filename)
    sysmat = sysmat_file.root.sysmat[:]

    # save_name = sysmat_filename[:-3] + '_interp'
    interp_sysmat = interpolate_system_response(sysmat, x_img_pixels=x_dim)  # save_fname=save_name)
    # sysmat_file.close()
    return interp_sysmat, sysmat_file  # to help close if need be


def interpolate_system_response(sysmat, x_img_pixels=75):  # needed for system_matrix_interpolate
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
    # np.save(save_fname, interp_sysmat)
    return interp_sysmat


# ===============Smoothing ================
def smooth_point_response(sysmat, x_img_pixels, *args, **kwargs):  # h5file = True

    size = args[0]
    try:
        fwhm = kwargs['fwhm']
    except Exception as e:
        print(e)
        print("Default FWHM used: 1")
        fwhm = 1

    print("Sysmat Shape:", sysmat.shape)

    fstr = str(int(fwhm)) + "_" + "{:.1f}".format(fwhm)[2:]
    return gaussian_smooth_response(sysmat, x_img_pixels, *args, **kwargs), "_F" + fstr + "S" + str(size)


def gaussian_smooth_response(sysmat, x_img_pixels, *args, **kwargs):  # needed for smooth_point_response
    # assumption is that sysmat shape is (n_pixels, n_measurements) i.e. (1875, 2304)
    tot_img_pixels, tot_det_pixels = sysmat.shape  # n_pixels, n_measurements

    view = sysmat.T.reshape([tot_det_pixels, tot_img_pixels // x_img_pixels,  x_img_pixels])
    # TODO: Might not need to transpose in this way
    smoothed_reponse = np.copy(view)
    print("View shape: ", view.shape)

    kern = make_gaussian(*args, **kwargs)  # size, fwhm=1
    ksize = kern.shape[0]
    # print("Kern: ", kern)
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


def make_gaussian(size, fwhm=1):  # f, center=None):  # needed for gaussian_smooth_response
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


# ===============Appending ================
def append_responses(files, save_name='appended'):  # sysmat files
    """Append responses that are adjacent in the second dimension"""
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


def append_FoVs(files, save_name='appended', first_dim_pxls=(101, 101), after=True):  # TODO: Rewrite this
    """Append responses that are adjacent in the first dimension. After means each file is appended after previous.
    first_dim_pxls is the number of pixels in appended direction for each file"""

    tmp_list = list(range(len(files)))
    second_dim_pxls = list(range(len(files)))  # should all be the same, double check
    tot_pxls = 0
    det_pxls = 0
    meas_pxls = 0

    for fid, file in enumerate(files):
        if tables.is_hdf5_file(file):
            sysmat_file = load_h5file(file)
            arr = sysmat_file.root.sysmat[:]
            sysmat_file.close()
        else:
            arr = np.load(file)
        meas_pxls, det_pxls = arr.shape
        tot_pxls += meas_pxls
        second_dim_pxls[fid] = meas_pxls//first_dim_pxls[fid]
        print("det_pxl: ", det_pxls)
        print("second_dim_pxls: ", second_dim_pxls[fid])
        print("first dim pxls: ", first_dim_pxls[fid])
        tmp_list[fid] = arr.T.reshape([det_pxls, second_dim_pxls[fid], first_dim_pxls[fid]])

        print("File {f} shape: {s}".format(f=fid, s=tmp_list[fid].shape))

    assert np.all(np.array(second_dim_pxls) == second_dim_pxls[0]), "Files don't have same shape in second dimension"
    if after:
        tot_arr = np.concatenate(tmp_list, axis=2)
    else:
        tot_arr = np.concatenate(tmp_list[::-1], axis=2)

    reshaped_arr = tot_arr.transpose((1, 2, 0)).reshape([tot_pxls, det_pxls])
    np.save(save_name, reshaped_arr)
    print("Final shape: ", reshaped_arr.shape)  # TODO: Test this with table measurements


# ===============Other ================
def load_h5file(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def sysmat_processing(files, npix, *args, interp=True, smooth=True, fname='processed', **kwargs):
    """*args = sze of gaussian filter and **kwargs is fhwm of filter. Files is list of sysmat_files and npix is list
    of dimensions of the spaces"""

    if isinstance(files, str):
        files = [files]

    if len(files) > np.array(npix).size//2:
        print("Checked")
        npix = [npix] * len(files)

    if len(files) == 1:
        npix = [npix]

    print("npix length: ", len(npix))
    print("npix: ", npix)
    print("first entry: ", npix[0])
    store_list = []
    append_str = ''

    for fid, file in enumerate(files):
        npix_x, npix_y = npix[fid]
        if interp:
            sysmat, sysmat_file = system_matrix_interpolate(file, x_dim=npix_x)
            # print("Sysmat shape now:", sysmat.shape)
            npix_x = npix_x + (npix_x - 1)
            npix_y = npix_y + (npix_y - 1)
            sysmat_file.close()
        else:
            if tables.is_hdf5_file(file):
                sysmat_file = load_h5file(file)
                sysmat = sysmat_file.root.sysmat[:]
                sysmat_file.close()  # TODO: Will this cause problems? Copy otherwise
            else:
                sysmat = np.load(file)  # .npy
        print("Interpolation successful!")
        print("sysmat.shape: ", sysmat.shape)
        if smooth:
            sysmat, append_str = smooth_point_response(sysmat, npix_x, *args, **kwargs)

        store_list.append(sysmat)

    fname += append_str
    # if len(store_list) == 1:
    #   processed_array = store_list[0]
    # else:
    processed_array = np.vstack(store_list)
    print("Final shape: ", processed_array.shape)
    np.save(fname, processed_array)

    # TODO: np.savez of system matrix, npix, and total_shape and integrate into recon function workflow
    # smooth_point_response("/home/justin/repos/sysmat/design/2021-03-30-2347_SP0_interp.npy", 201, 7,
    #                       h5file=False, fwhm=2.355 * 1)  # 2.355 * spread defined in gaussian function (uncertainty)


def main():
    # files = ['/home/justin/repos/sysmat/design/2021-04-01-2021_SP0.h5',  # 120 cm from source to collimator
    #          '/home/justin/repos/sysmat/design/2021-04-02-0012_SP0.h5',  # 110 cm
    #          '/home/justin/repos/sysmat/design/2021-04-02-0308_SP0.h5',  # 100
    #          '/home/justin/repos/sysmat/design/2021-04-02-1407_SP0.h5']  # 90
    # npix = np.array([37, 31])  # 3D
    # append_responses(files, save_name='fov3d')

    files = ['/home/justin/repos/sysmat/design/Apr28_FoV_F0_7S7.npy',
             '/home/justin/repos/sysmat/design/2021-04-23-1259_SP1.h5']
    append_responses(files, save_name="Apr28_FoV_beamstop")


def main2():  # Responses Aug 3, see tools.py in sysmat_current for original
    # FoV, Top FoV, Bot FoV, Table, Beam Stop, Beam Port
    region_files = [  # '/home/justin/repos/sysmat_current/sysmat/design/2021-07-21-1244_SP1.h5',  # FOV, subsample 1
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-07-23-1034_SP2.h5',  # FOV, subsample 2
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-07-22-0244_SP1.h5',  # Top FOV
                    '/home/justin/repos/sysmat_current/sysmat/design/2021-07-22-0957_SP1.h5'] # ,  # Bot FOV
                    # '/home/justin/repos/sysmat_current/sysmat/design/2021-07-22-2255_SP1.h5',  # Table
                    # '/home/justin/repos/sysmat_current/sysmat/design/2021-07-22-1348_SP1.h5',  # Beam Stop
                    # '/home/justin/repos/sysmat_current/sysmat/design/2021-07-22-1817_SP1.h5']  # Beam Port
    append_responses(region_files, save_name="aug3_just_FoV_s2")  # folded


if __name__ == "__main__":
    main2()

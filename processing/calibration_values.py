import numpy as np

# This is a placeholder for the moment. This should really just load from a text file. Used by one_module_processing
# NOTE (IMPORTANT): This is for beam on


def load_calibration(place):
    if place == 'Davis':  # place == 'Davis'
        return load_davis()
    else:
        return load_berkeley()


def load_davis():
    # All values in relation to a universal labeling scheme where, when facing the array from the front,
    # module 0 is the top left, module 3 is top right, module 4 is next row below module 0, etc. Module 15 is
    # the bottom right.
    # For PMTs, it is similar. Upper left is 0, upper right is 1, lower left is 2, lower right is 3

    calib = {}
    calib['pmts'] = np.array([[1, 3], [0, 2]])  # this is set when you plug in the cables
    # as looked at from the front how do the channels map to this 2x2 PMT grid i.e. where is the upper left,
    # upper right (first row), lower left, and lower right (second row) in terms of channels 1-4 on the first card
    calib['swapped'] = np.array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])  # index 6 is swapped
    calib['swapped_pmts'] = np.array([[0, 3], [1, 2]])   # np.array([[1, 2], [0, 3]]) or prob np.array([[0, 3], [1, 2]])
    # Swapped means channels got swapped. This should not happen normally
    calib['pmt_gains'] = np.array([[0.94, 1., 1., 0.97],  # 0
                                   [0.92, 1., 1.02, 1.],
                                   [0.94, 1., 1., 1.04],
                                   [1., 1., 1., 1.],
                                   [0.96, 1., 1., 1.],  # 4
                                   [0.97, 1.03, 1.03, 1.],
                                   [1.01, 1.02, 1.04, 1.],
                                   [1., 1.02, 0.99, 1.06],
                                   [1.08, 1.11, 1., 1.16],  # 8, last entry might be 1.18
                                   [0.93, 1., 0.97, 1.],
                                   [0.99, 0.98, 0.98, 1.],
                                   [1., 1., 0.95, 0.97],
                                   [0.94, 1., 0.92, 0.9],  # 12
                                   [0.89, 0.99, 0.93, 1.],  # [1., 0.5, 1., 1.],
                                   [1., 0.5 * 1.16, 1.04, 1.09],  # [1., 1., 1., 1.],  # TODO: Here
                                   [0.9, 1.03, 0.94, 1.]])

    calib['pmt_shifts'] = np.array([[0., 0., 0., 0.],  # 0
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],  # 4
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],  # 8
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],  # 12
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.]])

    calib['module_gains'] = np.array([1.02, 0.96, 1., 0.99,
                                      1., 0.98, 0.95, 1.02,
                                      0.89, 1.03, 1.02, 0.996,
                                      1.066, 1.07, 0.97, 1.04])
    calib['module_shifts'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # E(MeV) = A * log(ADC_value) - B
    calib['energy_a'] = np.array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])
    calib['energy_b'] = np.array([17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5])

    # The proper way to do this
    calib['alpha_undistort'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 0.])

    # TODO: Fix everything relative to beam on. Only 12, 13 done so far
    calib['crystal_x_edges'] = np.array([[0, 14.5, 21.5, 28, 33.5, 41, 50, 58.5, 65.5, 72, 79, 85.5, 100],  # 0
                                         [0, 15, 21.5, 28, 35, 41, 50, 57.5, 64.5, 70, 76, 82, 100],
                                         [0, 14, 20.5, 27, 33, 40, 49, 58, 65, 72, 78.5, 85, 100],
                                         [0, 13.5, 20.5, 27, 32.5, 40, 49, 58.5, 65.5, 72.25, 79, 86, 100],
                                         [0, 14.5, 21, 27.5, 33.5, 41.5, 50, 58.5, 65.5, 72, 79, 85.5, 100],  # 4
                                         [0, 14, 20.5, 27, 33, 41, 50, 58, 65, 72, 79, 85.5, 100],
                                         [0, 13.5, 20.5, 27, 33, 41, 50, 59, 67, 72.5, 79, 86, 100],
                                         [0, 13.5, 20.5, 27, 33, 40, 50, 58, 65, 72, 79, 86, 100],
                                         [0, 13.5, 20, 27, 32.5, 41, 49, 59, 67, 73, 79, 85.5, 100],  # 8
                                         [0, 14, 21, 27.5, 34, 41, 49, 58, 65, 71.5, 78.5, 85.5, 100],
                                         [0, 13., 19., 24, 30.5, 36, 46, 55, 63, 69, 76.5, 83, 100],
                                         [0, 14, 20.5, 27, 33.5, 41, 50.5, 59, 66.5, 73, 79.5, 86.5, 100],
                                         [0, 17.5, 24, 31, 37, 43, 51.5, 60.5, 67.5, 73, 78.5, 84, 100],  # 12 (Here)
                                         [0, 18.5, 25, 31.5, 36, 43, 50.5, 58, 65.5, 70, 76, 82.5, 100],
                                         [0, 14, 21, 27.5, 33.5, 41, 49.5, 58, 65, 72, 79, 86, 100],
                                         [0, 14.5, 21, 28.5, 35, 42, 51, 60, 66.5, 72, 79, 85.5, 100]])

    calib['crystal_y_edges'] = np.array([[0, 14.5, 21, 27.5, 33.5, 41, 49.5, 58.5, 65.5, 72, 79, 86, 100],  # 0
                                         [0, 14.5, 20.5, 27, 32.5, 40, 47, 55, 62.5, 69, 75, 81.5, 100],  #
                                         [0, 14.5, 20, 27, 33.5, 40.5, 49.5, 58, 65, 72, 78.5, 85, 100],
                                         [0, 13.5, 20, 26.5, 33, 40, 49, 58, 65.5, 72, 79, 86, 100],
                                         [0, 14.5, 21, 27.5, 33.5, 41.5, 49.5, 58, 65.5, 72, 79, 85.5, 100],  # 4
                                         [0, 14.5, 21, 27.5, 33.5, 41.5, 50, 58.5, 66, 72.5, 79, 86, 100],
                                         [0, 14, 20.5, 27, 33, 41, 50, 59, 66, 72.5, 79.5, 86, 100],
                                         [0, 14.5, 21, 27, 33, 40, 50, 58.5, 67., 73, 79.5, 86, 100],
                                         [0, 14.5, 19.5, 26, 33, 39.5, 48.5, 58, 65.5, 72, 78.5, 84.5, 100],  # 8
                                         [0, 14.5, 21, 27, 33.5, 40.5, 49.5, 58, 65.5, 72, 78.5, 85.5, 100],
                                         [0, 14.5, 20, 25.5, 31.5, 39.5, 48.5, 58, 65, 71.5, 77.5, 83, 100],
                                         [0, 14.5, 21.5, 27.5, 35, 42, 51.5, 61, 68, 74, 81, 86.5, 100],
                                         [0, 17., 24, 30, 35.5, 43, 52, 60.5, 67.5, 72.5, 79, 84, 100],  # 12
                                         [0, 18, 24, 29, 34, 41, 49, 57, 64, 69, 75, 81, 100],
                                         [0, 14, 21, 27, 33.5, 41.5, 50, 58, 66, 72, 79, 86, 100],
                                         [0, 15, 21.5, 28, 34, 42, 50.5, 58.5, 66, 72, 79, 85.5, 100]])

    return calib


def load_berkeley():
    # All values in relation to a universal labeling scheme where, when facing the array from the front,
    # module 0 is the top left, module 3 is top right, module 4 is next row below module 0, etc. Module 15 is
    # the bottom right.
    # For PMTs, it is similar. Upper left is 0, upper right is 1, lower left is 2, lower right is 3

    calib = {}
    calib['pmts'] = np.array([[1, 3], [0, 2]])  # this is set when you plug in the cables
    # as looked at from the front how do the channels map to this 2x2 PMT grid i.e. where is the upper left,
    # upper right (first row), lower left, and lower right (second row) in terms of channels 1-4 on the first card
    calib['swapped'] = np.array([0., 0., 0., 0., 0., 0., 0, 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    calib['swapped_pmts'] = np.array([[0, 3], [1, 2]])  # np.array([[1, 2], [0, 3]]) or np.array([[0, 3], [1, 2]])
    # Swapped means channels got swapped. This should not happen normally
    calib['pmt_gains'] = np.array([[1., 1., 1., 1.],  # 0
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],  # 4
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],  # 8
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],  # 12
                                   [1., 1., 1., 1.],
                                   [1., 0.5, 1., 1.],
                                   [1., 1., 1., 1.]])

    calib['pmt_shifts'] = np.array([[0., 0., 0., 0.],  # 0
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],  # 4
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],  # 8
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],  # 12
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.]])

    calib['module_gains'] = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    calib['module_shifts'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # E(MeV) = A * log(ADC_value) - B
    calib['energy_a'] = np.array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])
    calib['energy_b'] = np.array([17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5])

    # The proper way to do this
    calib['alpha_undistort'] = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 0.])

    calib['crystal_x_edges'] = np.array([[0, 14.5, 21.5, 28, 33.5, 41, 50, 58.5, 65.5, 72, 79, 85.5, 100],  # 0
                                         [0, 15, 21.5, 28, 35, 41, 50, 57.5, 64.5, 70, 76, 82, 100],
                                         [0, 14, 20.5, 27, 33, 40, 49, 58, 65, 72, 78.5, 85, 100],
                                         [0, 13.5, 20.5, 27, 32.5, 40, 49, 58.5, 65.5, 72.25, 79, 86, 100],
                                         [0, 14.5, 21, 27.5, 33.5, 41.5, 50, 58.5, 65.5, 72, 79, 85.5, 100],  # 4
                                         [0, 14, 20.5, 27, 33, 41, 50, 58, 65, 72, 79, 85.5, 100],
                                         [0, 13.5, 20.5, 27, 33, 41, 50, 59, 67, 72.5, 79, 86, 100],
                                         [0, 13.5, 20.5, 27, 33, 40, 50, 58, 65, 72, 79, 86, 100],
                                         [0, 13.5, 20, 27, 32.5, 41, 49, 59, 67, 73, 79, 85.5, 100],  # 8
                                         [0, 14, 21, 27.5, 34, 41, 49, 58, 65, 71.5, 78.5, 85.5, 100],
                                         [0, 13., 19., 24, 30.5, 36, 46, 55, 63, 69, 76.5, 83, 100],
                                         [0, 14, 20.5, 27, 33.5, 41, 50.5, 59, 66.5, 73, 79.5, 86.5, 100],
                                         [0, 14, 20.5, 27.5, 35, 42.5, 51, 60.5, 68, 73, 79.5, 86.5, 100],
                                         [0, 14, 20.5, 27, 32.5, 40.5, 50, 58.5, 66, 72, 79, 86, 100],
                                         [0, 14, 21, 27.5, 33.5, 41, 49.5, 58, 65, 72, 79, 86, 100],
                                         [0, 14.5, 21, 28.5, 35, 42, 51, 60, 66.5, 72, 79, 85.5, 100]])

    calib['crystal_y_edges'] = np.array([[0, 14.5, 21, 27.5, 33.5, 41, 49.5, 58.5, 65.5, 72, 79, 86, 100],  # 0
                                         [0, 14.5, 20.5, 27, 32.5, 40, 47, 55, 62.5, 69, 75, 81.5, 100],  #
                                         [0, 14.5, 20, 27, 33.5, 40.5, 49.5, 58, 65, 72, 78.5, 85, 100],
                                         [0, 13.5, 20, 26.5, 33, 40, 49, 58, 65.5, 72, 79, 86, 100],
                                         [0, 14.5, 21, 27.5, 33.5, 41.5, 49.5, 58, 65.5, 72, 79, 85.5, 100],  # 4
                                         [0, 14.5, 21, 27.5, 33.5, 41.5, 50, 58.5, 66, 72.5, 79, 86, 100],
                                         [0, 14, 20.5, 27, 33, 41, 50, 59, 66, 72.5, 79.5, 86, 100],
                                         [0, 14.5, 21, 27, 33, 40, 50, 58.5, 67., 73, 79.5, 86, 100],
                                         [0, 14.5, 19.5, 26, 33, 39.5, 48.5, 58, 65.5, 72, 78.5, 84.5, 100],  # 8
                                         [0, 14.5, 21, 27, 33.5, 40.5, 49.5, 58, 65.5, 72, 78.5, 85.5, 100],
                                         [0, 14.5, 20, 25.5, 31.5, 39.5, 48.5, 58, 65, 71.5, 77.5, 83, 100],
                                         [0, 14.5, 21.5, 27.5, 35, 42, 51.5, 61, 68, 74, 81, 86.5, 100],
                                         [0, 15., 21.5, 28, 34, 42.5, 51.5, 60.5, 67.5, 74, 80.5, 87, 100],
                                         [0, 13, 20, 26.5, 32.25, 39.5, 48.5, 58, 64.5, 71, 78.5, 85, 100],
                                         [0, 14, 21, 27, 33.5, 41.5, 50, 58, 66, 72, 79, 86, 100],
                                         [0, 15, 21.5, 28, 34, 42, 50.5, 58.5, 66, 72, 79, 85.5, 100]])
    return calib

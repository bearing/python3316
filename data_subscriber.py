import os

_supported_ftype = ('binary', 'hdf5')


def daq_readout(save=False, file_type='binary', output_dir=None, quiet=True, print_stats=False):
    if output_dir is None:
        output_dir = os.getcwd() + '/data'
    makedirs(output_dir)

    if file_type not in _supported_ftype:
        raise ValueError("Data type {d} is not one of: {f}".format(d=file_type, f=_supported_ftype))
    pass


def makedirs(path):
    """ Create directories for `path` (like 'mkdir -p'). """
    if not path:
        return
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

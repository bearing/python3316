import os
import sis3316_eth as dev


class daq_system(object):
    _supported_ftype = ('binary', 'hdf5')
    _ports = (6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6322)  # Hopefully this
    #  range stays unused

    def __init__(self, hostnames=None, configs=None, synchronize=False):
        if hostnames is None:  # TODO: Automatically generate hostnames from printed hardware IDs
            raise ValueError('Need to specify module ips!')
        if configs is None:
            raise ValueError('Need to specify config files!')
        assert len(hostnames) == len(configs), "You specified {c} configs for" \
                                               " {h} modules!".format(c=len(configs), h=len(hostnames))
        self.synchronize = synchronize
        self.configs = configs
        self.modules = [dev.Sis3316(mod_ip, port=port_num) for mod_ip, port_num in zip(hostnames, self._ports)]

    def __del__(self):
        for mod in self.modules:
            mod.close()

    def setup(self):
        # mods = iter(self.modules)
        if self.synchronize:
            self.modules[0].open()  # Enable ethernet communication
            self.modules[0].set_config(fname=self.configs[0], FP_LVDS_Master=int(True))  # The first module is assumed
            #  to be the master clock
            mods = iter(self.modules)
            next(mods)
            for ind, board in enumerate(mods):
                board.open()
                board.set_config(fname=self.configs[ind], FP_LVDS_Master=int(False))
            return

        for ind, board in enumerate(self.modules):
            board.open()
            board.set_config(fname=self.configs[ind])


def daq_readout(save=False, file_type='binary', output_dir=None, quiet=True, print_stats=False):
    if output_dir is None:
        output_dir = os.getcwd() + '/data'
    makedirs(output_dir)

    if file_type not in ('binary', 'hdf5'):
        raise ValueError("Data type {d} is not one of: {f}".format(d=file_type, f=('binary', 'hdf5')))
    pass


def makedirs(path):
    """ Create directories for `path` (like 'mkdir -p'). """
    if not path:
        return
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

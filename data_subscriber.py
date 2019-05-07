import os
import sis3316_eth as dev
from timeit import default_timer as timer
from datetime import datetime


class daq_system(object):
    _supported_ftype = {'binary': '.bin',
                        'hdf5': '.h5'}
    _ports = (6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6312)  # Hopefully this
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
        self.run = None

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
            for ind, board in enumerate(mods, start=1):
                board.open()
                board.configure(id=ind * 12)
                board.set_config(fname=self.configs[ind], FP_LVDS_Master=int(False))
            return

        for ind, board in enumerate(self.modules):
            board.open()
            board.configure(id=ind * 12)
            board.set_config(fname=self.configs[ind])

    def subscribe(self, max_time=60, gen_time=None, save_type='binary', save_fname=None):
        # Maybe add option to change save name?
        if save_type not in self._supported_ftype:
            raise ValueError('File type {f} is not supported. '
                             'Supported file types: {sf}'.format(f=save_type, sf=str(self._supported_ftype))[1:-1])

        if gen_time is None:
            gen_time = max_time  # I.E. swap on memory flags instead of time

        time_elapsed = 0
        gen = 0  # Buffer readout 'generation'
        time_last = 0  # Last readout

        # TODO: Disarm, then arm, then timestamp clear
        try:
            if save_fname is None:
                save_fname = os.path.join(os.getcwd(), 'Data', datetime.now().strftime("%Y%m%d-%H%M")
                                          + self._supported_ftype[save_type])
            makedirs(save_fname)  # Create Data folder if none exists
            # initialize_save_file

            start_time = timer()
            while time_elapsed < max_time:
                time_elapsed = timer() - start_time
                buffer_swap_time = time_elapsed - time_last

                if self.synchronize:
                    polling_stat = self.modules[0]._readout_status()
                    memory_flag = polling_stat['FP_threshold_overrun']
                else:
                    polling_stats = [mod._readout_status() for mod in self.modules]
                    memory_flag = any([mem_flag['threshold_overrun'] for mem_flag in polling_stats])

                if buffer_swap_time > gen_time or memory_flag:
                    time_last = timer()
                    gen += 1

                    for mods in self.modules:
                        mods.mem_toggle()  # Swap, then read


            # TODO: Wait 500 ms

                    # self.readout_buffer()
                    # push to file

        except KeyboardInterrupt:
            pass
        # self.synchronize # Use to tell you whether multimodule or not
        # Then keep polling _readout_status in readout. When flag tripped, swap memory banks and readout previous banks
        # For each channel in a module with non zero event length, create a empty numpy array that is the size of
        # prev_bank. Then use np.from_buffer to fill it  with readout making sure to track bank for bank_read call.
        # If binary save, turn a copy of that numpy array into a binary array and write to file
        pass

    def subscribe_no_save(self, max_time=60, gen_time=None):
        pass

    def _configure_hdf5(self):
        pass

    def save_raw_only(self, max_time=None, gen_time=None):  # Don't parse, save to binary (diagnostic method)
        pass

    # dt = np.dtype(np.uint16)
    # dt = dt.newbyteorder('<')
    # np.frombuffer(a,dt)

    # try:
    #    while True:
    #        do_something()
    # except KeyboardInterrupt:
    #    pass


def makedirs(path):
    """ Create directories for `path` (like 'mkdir -p'). """
    if not path:
        return
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

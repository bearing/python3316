import os
import sis3316_eth_new as dev
# import sis3316_eth as dev
from readout import destination  # TODO: This is clumsy
from timeit import default_timer as timer
from datetime import datetime
import tables
import numpy as np
from common.utils import msleep
from io import IOBase


class daq_system(object):
    _supported_ftype = {'binary': '.bin',
                        'hdf5': '.h5'}
    _ports = (6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6312)  # Hopefully this
    #  range stays unused

    def __init__(self, hostnames=None, configs=None, synchronize=False):
        if hostnames is None:  # TODO: Automatically generate hostnames from printed hardware IDs
            raise ValueError('Need to specify module ips!')
        # TODO: check if hostnames is just a string, if so turn to list of 1 entry
        # print("Hostnames: ", hostnames)
        # print("Configs: ", configs)
        if configs is None:
            raise ValueError('Need to specify config files!')
        assert len(hostnames) == len(configs), "You specified {c} configs for" \
                                               " {h} modules!".format(c=len(configs), h=len(hostnames))
        self.synchronize = synchronize
        self.configs = configs
        self.modules = [dev.Sis3316(mod_ip, port=port_num) for mod_ip, port_num in zip(hostnames, self._ports)]
        # print("Self.modules: ", self.modules)
        self.run = None
        self.file = None
        self.fileset = False
        self.event_formats = None

    def __del__(self):
        for mod in self.modules:
            mod.close()

    def setup(self):
        # mods = iter(self.modules)
        if self.synchronize:
            self.modules[0].open()  # Enable ethernet communication
            self.modules[0].set_config(fname=self.configs[0], FP_LVDS_Master=int(True))  # The first module is assumed

            for ind, board in enumerate(self.modules, start=1):
                board.open()
                board.configure(c_id=ind * 16)
                board.set_config(fname=self.configs[ind], FP_LVDS_Master=int(False))
            return

        for ind, board in enumerate(self.modules):
            board.open()
            board.configure(c_id=ind * 0x10)  # 16
            board.set_config(fname=self.configs[ind])
            # board.set_raw_window(fname=self.configs[ind])

    def _setup_file(self, save_type='binary', save_fname=None):
        if save_type not in self._supported_ftype:
            raise ValueError('File type {f} is not supported. '
                             'Supported file types: {sf}'.format(f=save_type, sf=str(self._supported_ftype))[1:-1])
        if save_fname is None:
            save_fname = os.path.join(os.getcwd(), 'Data', datetime.now().strftime("%Y-%m-%d-%H%M")
                                      + self._supported_ftype[save_type])
        makedirs(save_fname)

        hit_stats = [channel.event_stats for mod in self.modules for channel in mod.chan]

        if save_type is 'binary':
            file = open(save_fname, 'wb')
        else:
            file = tables.open_file(save_fname, mode="w", title="Data file")
        self.fileset = True
        return file, hit_stats

    def _h5_file_setup(self, file, hit_fmts):
        """ Sets up file structure for hdf5 """
        # TODO: Check file is hdf5 file object?
        max_ch = len(self.modules) * 4
        ch_group = [None] * max_ch
        # TODO: ADD HDF5  Datatype Support (1/4/2020)
        for ind in np.arange(max_ch):
            ch_group[ind] = file.create_group("/", 'ch' + str(ind), 'Ch Data')
            if hit_fmts[ind]['raw_event_length'] > 0:  # These lengths are defined to 16 bit words (see channel.py)
                pass  # Add Raw Data Group
            if hit_fmts[ind]['maw_event_length'] > 0:
                pass  # Save MAW Data
            if bool(hit_fmts[ind]['acc1_flag']):
                pass  # Set up first accumulator flag data types
            if bool(hit_fmts[ind]['acc2_flag']):
                pass  # Set up second accumulator flag data types
            if bool(hit_fmts[ind]['maw_flag']):
                pass  # Set up data types for maw trigger values
            if bool(hit_fmts[ind]['maw_max_values']):
                pass  # set up data types for FIR Maw (energy) values
            pass

    def subscribe(self, max_time=60, gen_time=None, **kwargs):
        # Maybe add option to change save name?
        if not self.fileset:
            self.file, self._event_formats = self._setup_file(**kwargs)
            # TODO: Generate data types and set up hdf5 file

        if gen_time is None:
            gen_time = max_time  # I.E. swap on memory flags instead of time


        time_elapsed = 0
        gen = 0  # Buffer readout 'generation'
        time_last = 0  # Last readout

        for device in self.modules:
            device.disarm()
            device.arm()
            device.ts_clear()  # TODO: This must be changed if start/pause/stop functionality exists

        try:
            # data_buffer = [[] for i in range(16)]
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

                    data_buffer = [[] for _ in range(16)]

                    for mod_ind, mods in enumerate(self.modules):
                        # TODO: mod_ind is not used in case the data load is too high. 1 module is limited to max of 2
                        # GB of data.
                        for chan_ind, chan_obj in enumerate(mods.chan):
                            # data_buffer[chan_ind] = mods.readout_buffer(chan_obj) # TODO: This might be wrong
                            data_buffer[chan_ind] = mods.readout_buffer(chan_ind)

                msleep(500)  # wait 500 ms
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

    def save_raw_only(self, max_time=None, gen_time=None, **kwargs):  # Don't parse, save to binary (diagnostic method)
        # Maybe add option to change save name?
        if not self.fileset:
            self.file, self.event_formats = self._setup_file(**kwargs)

        # NOTE: Unique to saving only raw binaries
        # proxy_file_object = destination(self.file)
        proxy_file_object = self.file

        if max_time is None:
            max_time = 60
            Warning("Max acquisition time in seconds not specified. Defaulting to ", max_time, " seconds!")

        if gen_time is None:
            gen_time = max_time  # I.E. swap on memory flags instead of time

        time_elapsed = 0
        gen = 0  # Buffer readout 'generation'
        time_last = 0  # Last readout

        for device in self.modules:
            # device.configure()
            device.disarm()
            device.arm()
            device.ts_clear()  # TODO: This must be changed if start/pause/stop functionality exists
            device.mem_toggle()
            print("Initial Status: ", device.status)

        try:
            # data_buffer = [[] for i in range(16)]
            start_time = timer()
            while time_elapsed < max_time:
                time_elapsed = timer() - start_time
                buffer_swap_time = time_elapsed - time_last
                # print("Time elapsed: ", time_elapsed)

                if self.synchronize:
                    polling_stat = self.modules[0]._readout_status()
                    memory_flag = polling_stat['FP_threshold_overrun']
                else:
                    polling_stats = [mod._readout_status() for mod in self.modules]
                    memory_flag = any([mem_flag['threshold_overrun'] for mem_flag in polling_stats])

                # for device in self.modules:
                #    print("Readout status: ", device.status)

                if buffer_swap_time > gen_time or memory_flag:
                    time_last = timer()
                    gen += 1
                    for mods in self.modules:
                        mods.mem_toggle()  # Swap, then read

                    # data_buffer = [[] for i in range(16)]

                    for mod_ind, mods in enumerate(self.modules):
                        # TODO: mod_ind is not used in case the data load is too high. 1 module is limited to max of 2
                        # GB of data.
                        print()
                        print("Generation ", gen, " Readout:")
                        for chan_ind, chan_obj in enumerate(mods.chan):
                            # print("Channel ", chan_ind, " Actual Memory Address: ", chan_obj.addr_actual)
                            print("Channel ", chan_ind, " Previous Memory Address: ", chan_obj.addr_prev)
                            for ret in mods.readout(chan_ind, proxy_file_object):
                                print("Bytes Transferred: ", ret['transfered'] * 4)
                                if chan_obj.event_stats['event_length'] > 0:
                                    print("Events Recorded ", "(Channel ", chan_ind, "): ",
                                          (ret['transfered'] * 4 / (2 * chan_obj.event_stats['event_length'])))

                msleep(500)  # wait 500 ms

            gen += 1
            for mods in self.modules:
                mods.mem_toggle()

            print()
            print("Clean Up")
            for mod_ind, mods in enumerate(self.modules):  # Dump remaining data
                for chan_ind, chan_obj in enumerate(mods.chan):
                    # print("Channel ", chan_ind, " Actual Memory Address: ", chan_obj.addr_actual)
                    print("Channel ", chan_ind, " Previous Memory Address: ", chan_obj.addr_prev)
                    mods.readout(chan_ind, proxy_file_object)
                    for ret in mods.readout(chan_ind, proxy_file_object):
                        print("Bytes Transferred: ", ret['transfered'] * 4)

        except KeyboardInterrupt:
            pass

        self.file.close()

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


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('hosts', type=str, help='list of hostnames or IP addresses')
    # parser.add_argument('configs', type=str,  help='list of config file names (1 for each board)')
    # parser.add_argument('synchronize', type=bool, nargs="?", default=False, help='Use first host as master clock')
    # args = parser.parse_args()

    dsys = daq_system(hostnames=['192.168.1.11'],
                      # configs=['/Users/justinellin/repos/python_SIS3316/sample_configs/NSCtest.json'],
                      # configs=['/Users/justinellin/repos/python_SIS3316/sample_configs/PGItest.json'],
                      configs=['/Users/justinellin/repos/python_SIS3316/sample_configs/RadMaptest.json'],
                      synchronize=False)
    mod0 = dsys.modules[0]
    print("mod ID:", hex(mod0._read_link(0x4)))
    print("Hardware Version: ", hex(mod0.hardwareVersion))
    mod0.open()
    print("Temperature (Celsius): ", mod0.temp)
    print("Serial Number: ", mod0.serno)
    print("Attempting to Set Config")
    dsys.setup()

    print("Finished setting config values!")

    print("Reading back set values!")
    print()

    for gid, grp in enumerate(mod0.grp):
        # print("Trigger Gate Window Length Group", gid, ": ", grp.gate_window)
        print("=FPGA Group ", gid, "Values=")
        print("Header :", grp.header)
        print("Gate Window: ", grp.gate_window)
        print("Raw Samples (window): ", grp.raw_window)
        print("Raw Sample Start Index: ", grp.raw_start)
        print("Peak + Gap Extra Delay Enable: ", bool(grp.delay_extra_ena))
        print("Pile-up Window: ", grp.pileup_window)
        print("Repile-up Window: ", grp.repileup_window)
        print("MAW Window: ", grp.maw_window)
        print("MAW Delay : ", grp.maw_delay)
        print("Address Threshold (32 bit words): ", grp.addr_threshold)
        print("=Sum Trigger Settings=")
        print("Peaking Time (samples): ", mod0.sum_triggers[gid].maw_peaking_time)
        print("Gap Time : ", mod0.sum_triggers[gid].maw_gap_time)
        print("Single Trigger Values: ", mod0.trig[gid].threshold)
        print("Sum Threshold Value: ", mod0.sum_triggers[gid].threshold)
        print("Sum Trigger Enabled: ", bool(mod0.sum_triggers[gid].enable))
        print()
        # print("Pre-Trigger Delay: ", grp.delay)
        # print("Peak + Gap Extra Delay: ", bool(grp.delay_extra_ena))

    for cid, channel in enumerate(mod0.chan):
        print("=Channel ", cid, "Values=")
        print("Voltage Range (0: 5V, 1: 2V, 2: 1.9V): ", channel.gain)
        # print("DAC Offset: ", channel.dac_offset)
        print("Event Types: ", channel.flags)
        print("Event Flags: ", channel.format_flags)
        print("Event Types Set : ", np.array(channel.hit_flags)[np.array(channel.format_flags).astype(bool)])
        print("Hit/Event Data (16 bit words): ", channel.event_stats)
        print("Long Shaper (Energy) Peaking Time: ", channel.en_peaking_time)
        print("Long Shaper (Energy) Gap Time: ", channel.en_gap_time)
        print("=Trigger Settings=")
        print("Peaking Time: ", mod0.trig[cid].maw_peaking_time)
        print("Gap Time: ", mod0.trig[cid].maw_gap_time)
        print("Enabled: ", bool(mod0.trig[cid].enable))

        print()

    print("Frequency: ", mod0.freq)
    # print("Sanity Check: ", )
    # print()

    print("Attemping test run!")
    dsys.save_raw_only(max_time=5)

    # RAW_DATA_BUFFER_CONFIG_REG =       					                      0x20
    # SIS3316_FPGA_ADC_GRP_REG_BASE = 0x1000
    # SIS3316_FPGA_ADC_GRP_REG_OFFSET = 0x1000
    # for gid, grp in enumerate(mod0.grp):
    #     testreg = 0x20 + grp.idx * 0x1000 + 0x1000
    #     print("=Raw Samples Register for Group ", grp.idx, "=")
    #     data = mod0.read(testreg)
    #     print("Raw Sample Length: ", (data >> 16) & 0xFFFe)
    #     print("Raw Sample Start Index: ", data & 0xFFFe)


if __name__ == "__main__":
    # import argparse
    main()

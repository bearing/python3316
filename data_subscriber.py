import os
import sis3316_eth_new as dev
import processing.parser as on_the_fly
# import sis3316_eth as dev
from readout import destination  # TODO: This is clumsy
from timeit import default_timer as timer
from datetime import datetime
import tables
import numpy as np
from common.utils import msleep
from io import IOBase
from processing.h5file import h5f


class daq_system(object):
    _supported_ftype = {'binary': '.bin',
                        'hdf5': '.h5'}
    _ports = (6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6312)  # Hopefully this
    #  range stays unused

    def __init__(self, hostnames=None, configs=None, synchronize=False, save_data=False,
                 ts_clear=False, verbose=False):
        if hostnames is None:  # TODO: Automatically generate hostnames from printed hardware IDs
            raise ValueError('Need to specify module ips!')
        if isinstance(hostnames, str):
            hostnames = [hostnames]

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
        self.ts_clear = ts_clear
        self.save = save_data
        self.verbose = verbose

    def __del__(self):
        for mod in self.modules:
            mod.close()

    def setup(self):
        if self.synchronize:
            self.modules[0].open()  # Enable ethernet communication
            self.modules[0].set_config(fname=self.configs[0], FP_LVDS_Master=int(True))  # The first module is assumed

            for ind, board in enumerate(self.modules, start=1):
                board.open()
                board.set_config(fname=self.configs[ind], FP_LVDS_Master=int(False))
                board.configure(c_id=ind * 16)
            return

        for ind, board in enumerate(self.modules):
            board.open()
            board.set_config(fname=self.configs[ind])
            board.configure(c_id=ind * 0x10)  # 16

    def _setup_file(self, save_type='binary', save_fname=None, **kwargs):
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
            file = h5f(save_fname, hit_stats, **kwargs)
        self.fileset = True
        return file, hit_stats

    def subscribe_with_save(self, max_time=60, gen_time=None, **kwargs):
        if not self.fileset:
            self.file, self._event_formats = self._setup_file(**kwargs)

        if gen_time is None:
            gen_time = max_time  # I.E. swap on memory flags instead of time

        hit_parser = on_the_fly.parser(self.modules)

        time_elapsed = 0
        gen = 0  # Buffer readout 'generation'
        time_last = 0  # Last readout

        for device in self.modules:
            device.disarm()
            device.arm()
            if self.ts_clear:
                device.ts_clear()

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

                    for mod_ind, mods in enumerate(self.modules):
                        for chan_ind, chan_obj in enumerate(mods.chan):
                            tmp_buffer = mods.readout_buffer(chan_ind)
                            event_dict, evts = hit_parser.parse(tmp_buffer, mod_ind, chan_ind)
                            self.file.save(event_dict, evts, mod_ind, chan_ind)

                msleep(500)  # wait 500 ms

        except KeyboardInterrupt:
            for mod in self.modules:
                del mod

        if self.verbose:
            print("Finished!")

    def subscribe_no_save(self, max_time=60, gen_time=None, **kwargs):

        if gen_time is None:
            gen_time = max_time  # I.E. swap on memory flags instead of time

        hit_parser = on_the_fly.parser(self.modules)

        time_elapsed = 0
        gen = 0  # Buffer readout 'generation'
        time_last = 0  # Last readout

        for device in self.modules:
            device.disarm()
            device.arm()
            if self.ts_clear:
                device.ts_clear()

        try:
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

                    for mod_ind, mods in enumerate(self.modules):
                        for chan_ind, chan_obj in enumerate(mods.chan):
                            if self.verbose:
                                print("Channel ", chan_ind, " Actual Memory Address: ", chan_obj.addr_actual)
                                print("Channel ", chan_ind, " Previous Memory Address: ", chan_obj.addr_prev)
                            tmp_buffer = mods.readout_buffer(chan_ind)
                            event_dict, evts = hit_parser.parse(tmp_buffer, mod_ind, chan_ind)
                            print("Dictionary:", event_dict)

                msleep(500)  # wait 500 ms

        except KeyboardInterrupt:
            for mod in self.modules:
                del mod

        if self.verbose:
            print("Finished!")


    def save_raw_only(self, max_time=None, gen_time=None, **kwargs):  # Don't parse, save to binary (diagnostic method)
        if not self.fileset:
            self.file, self.event_formats = self._setup_file(**kwargs)

        # NOTE: Unique to saving only raw binaries
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
            device.configure()
            device.disarm()
            device.arm()
            device.ts_clear()
            device.mem_toggle()
            print("Initial Status: ", device.status)
            print("Beginning Readout")

        try:
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

                    for mod_ind, mods in enumerate(self.modules):
                        if self.verbose:
                            print()
                            print("Generation ", gen, " Readout:")
                        for chan_ind, chan_obj in enumerate(mods.chan):
                            if self.verbose:
                                print("Channel ", chan_ind, " Actual Memory Address: ", chan_obj.addr_actual)
                                print("Channel ", chan_ind, " Previous Memory Address: ", chan_obj.addr_prev)
                            for ret in mods.readout(chan_ind, proxy_file_object):
                                if self.verbose:
                                    print("Bytes Transferred: ", ret['transfered'] * 4)
                                if chan_obj.event_stats['event_length'] > 0:
                                    if self.verbose:
                                        print("Events Recorded ", "(Channel ", chan_ind, "): ",
                                              (ret['transfered'] * 4/ (2 * chan_obj.event_stats['event_length'])))
                msleep(500)  # wait 500 ms

            gen += 1
            for mods in self.modules:
                mods.mem_toggle()

            print()
            print("Clean Up")
            for mod_ind, mods in enumerate(self.modules):  # Dump remaining data
                for chan_ind, chan_obj in enumerate(mods.chan):
                    if self.verbose:
                        print("Channel ", chan_ind, " Actual Memory Address: ", chan_obj.addr_actual)
                        print("Channel ", chan_ind, " Previous Memory Address: ", chan_obj.addr_prev)
                    mods.readout(chan_ind, proxy_file_object)
                    for ret in mods.readout(chan_ind, proxy_file_object):
                        if self.verbose:
                            print("Bytes Transferred: ", ret['transfered'] * 4)
                        if chan_obj.event_stats['event_length'] > 0:
                            if self.verbose:
                                print("Events Recorded ", "(Channel ", chan_ind, "): ",
                                      (ret['transfered'] * 4 / (2 * chan_obj.event_stats['event_length'])))
            print("Finished!")

        except KeyboardInterrupt:
            self.file.close()

        self.file.close()


def makedirs(path):
    """ Create directories for `path` (like 'mkdir -p'). """
    if not path:
        return
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-f', nargs='+', required=True, help='input config file(s)')
    parser.add_argument('--ips', '-i', nargs='+', required=True, help='IP addresses of 3316 modules')
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose flag (prints to terminal)')
    # parser.add_argument('--hdf5', '-h5', action='store_true', help='save hit data as hdf5 file')
    parser.add_argument('--keep_config', '-k', action='store_true', help='set to keep current loaded configs')
    parser.add_argument('--ts_keep', '-t', action='store_false', help='set to not clear timestamps')
    # parser.add_argument('--binary', '-b', action='store_true', help='save hit data to binary')
    parser.add_argument('--gen_t', '-g', nargs=1, type=float, default=2,
                        help='Max time between reads in seconds (default is 2)')
    parser.add_argument('--save', '-s', nargs=1, choices=['raw_binary', 'raw_hdf5', 'recon_hdf5'],
                        help='raw binary: text file dump. raw_hdf5: save 3316 raw data to hdf5. '
                             'recon_hdf5: user provided (see docs)')
    args = parser.parse_args()

    files = args.files
    hosts = args.ips
    verbose = args.verbose  # boolean
    keep_config = args.keep_config
    ts_clear = args.ts_keep
    gen_time = args.gen_t
    save_option = args.save

    n_boards = len(hosts)
    n_configs = len(files)

    sync = (n_boards > 1)

    if n_configs is 1 and n_boards > 1:
        files = files * n_boards  # Copy config to every board

    dsys = daq_system(hostnames=hosts, configs=files, synchronize=sync, ts_clear=ts_clear, verbose=verbose)

    print("Number of Modules: ", len(dsys.modules))
    print("Keep Config?", keep_config)
    if not keep_config:
        print("Attempting to Set Config")
        dsys.setup()
        print("Finished setting config values!")

    if verbose:
        print("Reading back set values!")
        for ind, mod in enumerate(dsys.modules):
            print("==================================")
            print("mod {n} ID: {h}".format(n=ind, h=hex(mod._read_link(0x4))))
            print("Hardware Version: ", hex(mod.hardwareVersion))
            print("Temperature (Celsius): ", mod.temp)
            print("Serial Number: ", mod.serno)
            print("Frequency: ", mod.freq)

            for gid, grp in enumerate(mod.grp):
                print("=FPGA Group ", gid, "Values=")
                print("Firmware. Type:", grp.firmware_version['type'], ". Version:",
                      grp.firmware_version['version'], ". Revision:", grp.firmware_version['revision'])
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
                print("Peaking Time (samples): ", mod.sum_triggers[gid].maw_peaking_time)
                print("Gap Time : ", mod.sum_triggers[gid].maw_gap_time)
                print("Sum Threshold Value: ", mod.sum_triggers[gid].threshold)
                print("Sum Trigger Enabled: ", bool(mod.sum_triggers[gid].enable))
                print("Pre-Trigger Delay: ", grp.delay)
                print("Peak + Gap Extra Delay: ", bool(grp.delay_extra_ena))
                print()

            for cid, channel in enumerate(mod.chan):
                print("=Channel ", cid, "Values=")
                print("Voltage Range (0: 5V, 1: 2V, 2: 1.9V): ", channel.gain)
                # print("DAC Offset: ", channel.dac_offset)
                print("Termination Enabled (50 Ohm): ", channel.termination)
                print("Event Types: ", channel.flags)
                print("Event Flags: ", channel.format_flags)
                print("Event Types Set : ", np.array(channel.hit_flags)[np.array(channel.format_flags).astype(bool)])
                print("Hit/Event Data (16 bit words): ", channel.event_stats)
                print("Long Shaper (Energy) Peaking Time: ", channel.en_peaking_time)
                print("Long Shaper (Energy) Gap Time: ", channel.en_gap_time)
                print("=Trigger Settings=")
                print("Trigger Threshold:", mod.trig[cid].threshold)
                print("Peaking Time: ", mod.trig[cid].maw_peaking_time)
                print("Gap Time: ", mod.trig[cid].maw_gap_time)
                print("Enabled: ", bool(mod.trig[cid].enable))
                print()

    if save_option is 'raw_binary':
        dsys.save_raw_only(max_time=5)
    if save_option in ('raw_hdf5', 'recon_hdf5'):
        dsys.subscribe_with_save(gen_time=gen_time, max_time=2, data_save_type=save_option)
    else:
        dsys.subscribe_no_save(gen_time=gen_time, max_time=2)


if __name__ == "__main__":
    import argparse
    main()

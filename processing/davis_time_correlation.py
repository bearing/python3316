import tables
import io
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

# '/Users/justinellin/Desktop/Davis 2020/Tuesday/2020-10-06-1503.h5'
# -> beam axis is +x-hat, away from earth is +y-hat


class h5processed(object):
    def __init__(self, filepath, window):

        raw_samples = 26  # TODO: Check dynamically

        w1str = str(window[0] * 4)
        w2str = str(window[1] * 4)

        if window[0] >= 0:
            w1str = '+' + w1str

        if window[1] >= 0:
            w2str = '+' + w2str

        # TODO: Check if str?
        self.new_file_name = 'processedW' + w1str + w2str + '_' + filepath.replace('.h5', '')[-10:] + '.h5'
        # Should be just the date str
        self.lso_data_fields = ['bid', 'mod_id', 'ts', 'rel_ts', 'E1', 'E2', 'E3', 'E4']
        self.lso_data_types = [np.uint32, np.uint8, np.uint64, np.float32, np.uint32, np.uint32, np.uint32, np.uint32]
        self.file = tables.open_file(self.new_file_name, mode="w", title="Processed Data File")

        # Creating folder structure of new file
        lso_dtypes = np.dtype({"names": self.lso_data_fields, "formats": self.lso_data_types})
        self.file.create_table('/', 'event_data', description=lso_dtypes)
        self.file.create_earray('/', 'scin_raw', atom=tables.atom.UInt16Atom(), shape=(0, raw_samples))
        self.file.create_earray('/', 'scin_ts', atom=tables.atom.UInt16Atom(), shape=(0, 1))

    def save(self, data_dict, evts):
        if not evts:  # i.e. check for no events and then skip the rest
            return
        try:
            # print("Events:", evts)
            base_node = '/'

            # TODO: Check this works. If not have to check for each class type

            for table in self.file.iter_nodes(base_node, classname='Table'):  # Structured data sets
                # print("Table description:", table.description._v_dtype)
                data_struct = np.zeros(evts, dtype=table.description._v_dtype)
                for field in table.description._v_names:  # Field functions as a key
                    if data_dict[field] is not None:
                        # print("Field:", field)
                        # print("Data Dict[field]:", data_dict[field])
                        data_struct[field] = data_dict[field]
                table.append(data_struct)
                table.flush()

            for earray in self.file.iter_nodes(base_node, classname='EArray'):  # Homogeneous data sets
                earray.append(data_dict[earray.name])
                earray.flush()

            self.file.flush()

        except Exception as e:
            print(e)


class time_recon(object):
    _crystal_coordinates = 'peak_coords_mean.txt'

    def __init__(self, filepath, test=0, span=10):
        self.h5file = load_data(filepath)
        self.crd = load_coordinates(self._crystal_coordinates)
        self.pxl_mapper = pixel_mapper(self.crd)
        self.histogram_bins = np.arange(0, 100000, 1000)
        self.gamma_events = 0
        self.proton_events = 0
        self.module = 0
        self.lso_scan_idx = np.full(16, 0)  # These will track where current sweep ended so as to not waste time scanning
        # all proton bunches
        self.current_first_bunch = 0
        self.current_last_bunch = 0  # This will help determine when to stop checking nearest neighbor per LSO bunch
        self.current_num_bunches = 0
        self.current_bunch_start = 0

        self.test_mode = test
        # Defining Window

        if isinstance(span, int):
            window = np.array([-span//4, span//4])
        else:
            window = np.array(span//4).sort()

        if window.size > 2 or window.size < 1:
            raise ValueError('Window should be 1 value if symmetric or 2. '
                             'Size {s} provided'.format(s=np.array(window).size))
        self.window = window
        #

        self.processed_file = h5processed(filepath, self.window)
        print("Processed File set up!")

        self.lso_evts = [None] * 64
        self.lso_totals = [0 for i in np.arange(16)]  # [None] * 16, make PEP stop complaining
        for integer in np.arange(64):
            folder = '/det' + str(int(integer))
            node = self.h5file.get_node('/', folder).EventData
            self.lso_evts[integer] =  node
            if integer % 4 == 0:
                self.lso_totals[integer//4] = node.nrows

        scin_folder = '/det' + str(64)
        self.scin_evts = self.h5file.get_node('/', scin_folder).EventData
        self.scin_waveforms =  self.h5file.get_node('/', scin_folder).raw_data

        print('Total LSO Evts:', np.sum(np.array(self.lso_totals)))
        print('LSO events by module:', np.array(self.lso_totals))
        print('Total proton evts:', self.scin_evts.nrows)

        if test:  # i.e. test is non-zero
            self.proton_events = test
            print('Test Mode: Maximum number of bunches to sift through:', test)
        else:
            self.proton_events = self.scin_evts.nrows

        self.lso_data_fields = ['bid', 'mod_id', 'ts', 'rel_ts', 'E1', 'E2', 'E3', 'E4']
        self.lso_data_types = [np.uint32, np.uint8, np.uint64, np.float32, np.uint32, np.uint32, np.uint32, np.uint32]
        self.r3_data_types = np.dtype({"names": self.lso_data_fields[1:], "formats": self.lso_data_types[1:]})
        # print('R3 data types:', self.r3_data_types)

        # scintillator fields: scin_raw, scin_ts

        self.chunk_size = 100000  # number of proton bunches at 1 time
        self.scan_size = 100000  # Number of LSO events to scan at 1 time
        self.bid_global_corr = 0  # This helps to track where we are in numbers of correlated proton bunch events
        # self.histogram_bins = np.linspace(0, 100000, 3000)

        # Temporary storage for each sweep of a module
        self.correlated_bunch_indices = np.zeros(self.chunk_size)
        self.correlated_gamma_indices = np.zeros(self.chunk_size)
        # Running temporary memory bank, at most 20k evts per sweep

    def time_correlate(self):

        mod_idx = np.arange(64//4)  # 16
        mod_bid_store = [None] * 16  # Temporary store correlated proton bunch IDs
        mod_gid_store = [None] * 16  # Temporary store of correlated gamma IDs
        mod_evt_data_store = [None] * 16

        # scin_channel = 64  # channel 65

        # scin_folder = '/det' + str(scin_channel)
        # self.scin_evts = self.h5file.get_node('/', scin_folder).EventData
        scin_timestamps = self.scin_evts.col('timestamp')

        print('Total time measured in seconds:', (scin_timestamps[-1] - scin_timestamps[0]) * 4/10**9)
        process = True
        blk_ind = 0
        # chunk = 20000
        chunk = self.chunk_size

        # for integer in lso_channels:
        #    folder = '/det' + str(int(integer))
        #    self.lso_evts[integer] =  self.h5file.get_node('/', folder).EventData

        while process:
            start = blk_ind * chunk
            last_evt = (blk_ind + 1) * chunk
            # print('Start:', start)
            # print('Last_evt:', last_evt)
            # print('Bunches', self.proton_events)
            self.current_bunch_start = start
            if last_evt <= self.proton_events:
                current_protons = scin_timestamps[start:last_evt]
                blk_ind += 1
            else:
                current_protons = scin_timestamps[start:self.proton_events]
                process = False
            print('Percent Processed: {p}'.format(p=str(100.0 * start/self.proton_events)))

            if current_protons.size == 0:  # Prevents weird errors if multiple of chunk size for bunches
                continue

            self.current_first_bunch = current_protons[0]
            self.current_last_bunch = current_protons[-1]
            self.current_num_bunches = current_protons.size
            # print('First bunch: ', self.current_first_bunch)
            # print('Last bunch: ', self.current_last_bunch)
            # event_multiplicity = np.zeros(self.current_num_bunches)
            # ref = spatial.cKDTree(current_protons)  # bunches
            ref = spatial.cKDTree(current_protons[:, None])  # bunches

            for mod_id in mod_idx:
                self.module = mod_id
                mod_ts = self.lso_evts[mod_id * 4].col('timestamp')
                # print('mod_ts.shape:', mod_ts.shape)

                mod_bid_store[mod_id], mod_gid_store[mod_id], mod_evt_data_store[mod_id] =\
                    self._time_correlate_module(ref, mod_ts, self.window)

            if all(bids is None for bids in mod_bid_store):  # catch for NO events
                continue
            dict, evts = self._create_dictionary(mod_bid_store, mod_gid_store, mod_evt_data_store, start)
            self.processed_file.save(dict, evts)
        print("Finished!")

    def _create_dictionary(self, bid_store, gid_store, mod_evt_store, start):
        # start is the current global index of the proton events
        dict = {}

        # self.lso_data_fields = ['bid', 'mod_id', 'rel_ts', 'E1', 'E2', 'E3', 'E4']
        # print('bid store:', bid_store)
        bids = np.concatenate([bids for bids in bid_store if bids is not None])  # all correlated bunch ids of current sweep
        sorted_evt_inds = np.argsort(bids)
        evts = bids.size

        bids_hist = np.bincount(bids, minlength=self.chunk_size)

        proton_raw_ids = start + np.flatnonzero(bids_hist)
        dict['scin_raw'] = self.scin_waveforms[proton_raw_ids, :]  # Should be rows. Maybe [p_id,:]?
        dict['scin_ts'] = self.scin_evts.col('timestamp')[proton_raw_ids][:, None]

        num_new_correlated_bids = np.count_nonzero(bids_hist)
        multiplicity = bids_hist[bids_hist > 0]

        gamma_bunch_ids = self.bid_global_corr + np.arange(num_new_correlated_bids)
        dict['bid'] = np.repeat(gamma_bunch_ids, multiplicity)
        self.bid_global_corr += num_new_correlated_bids

        gamma_evts = np.concatenate([mod_evts for mod_evts in mod_evt_store if mod_evts is not None])[sorted_evt_inds]
        # dict['mod_id'] = gamma_evts[:, 0]
        # dict['ts'] = gamma_evts[:, 1]  # Datatype might be too large?
        # dict['rel_ts'] = gamma_evts[:, 2]
        # dict['E1'] = gamma_evts[:, 3]
        # dict['E2'] = gamma_evts[:, 4]
        # dict['E3'] = gamma_evts[:, 5]
        # dict['E4'] = gamma_evts[:, 6]
        for name in gamma_evts.dtype.names:
            dict[name] = gamma_evts[name]
        #for key, value in gamma_evts.items():
        #    dict[key] = value
        # dict['mod_id'] = gamma_evts['mod_id']
        # dict['ts'] = gamma_evts['ts']
        # dict['rel_ts'] = gamma_evts['rel_ts']
        # dict['E1'] = gamma_evts['E1']
        # dict['E2'] = gamma_evts['E2']
        # dict['E3'] = gamma_evts['E3']
        # dict['E4'] = gamma_evts['E4']
        return dict, evts

    def _time_correlate_module(self, scin_ts_tree, mod_ts, window):
        subprocess = True
        prev_end = self.lso_scan_idx[self.module]  # last LSO event in previous scan
        scan_block = 0
        chunk = self.scan_size
        tot = self.lso_totals[self.module]
        # correlated_indices = np.array([], dtype=np.uint64)
        correlated_evts = 0
        one_shift = True  # This will change only once per scan
        idx_shift = 0

        while subprocess:
            start = prev_end + (scan_block * chunk)  # earliest LSO event
            # last_evt = start + ((scan_block + 1) * chunk)  # latest LSO event. This was wrong
            last_evt = start + chunk  # latest LSO event

            if start > tot:
                break

            # print('start:', start)
            if mod_ts[start] > self.current_last_bunch:  # lso scan has  bypassed the last current bunch ts
                # subprocess = False
                break

            if last_evt >= tot:
                last_evt = tot - 1
                current_gamma = mod_ts[start:, None]
                subprocess = False
            else:
                current_gamma = mod_ts[start:last_evt, None]
                scan_block += 1

            # if mod_ts[last_evt] < self.current_first_bunch:  # lso scan is "too early"
            if mod_ts[last_evt] < self.current_first_bunch:
                scan_block += 1
                continue

            if mod_ts[last_evt] > self.current_last_bunch and one_shift:
                # prev_end = start
                # self.lso_scan_idx[self.module] = start
                idx_shift = start
                one_shift = False  # No more moving the start point
                # I.E. the current lso events passed the end of the current proton event block.
                # Since it always increases this means the next go around you can start there since the start of this
                # iteration will be right before the switch

            # All of this processing is for these next few steps. Geez.
            dist, idx = scin_ts_tree.query(current_gamma)
            cor_idx = (dist > window[0]) & (dist < window[1])
            cor_gamma = start + np.flatnonzero(cor_idx)  # gamma events matching a proton bunch. Will not repeat
            cor_proton = idx[(dist > window[0]) & (dist < window[1])]  # bunch ids. May repeat

            common_evts = cor_proton.size  # new  events
            if common_evts == 0:
                continue
            self.correlated_gamma_indices[correlated_evts:(correlated_evts + common_evts)] = cor_gamma
            # self.correlated_bunch_indices[correlated_evts:common_evts] = cor_proton
            self.correlated_bunch_indices[correlated_evts:(correlated_evts + common_evts)] = cor_proton
            correlated_evts += common_evts

        if idx_shift:
            self.lso_scan_idx[self.module] = idx_shift

        if correlated_evts == 0:
            return None, None, None

        r1 = self.correlated_bunch_indices[:correlated_evts].astype(int)
        r2 = self.correlated_gamma_indices[:correlated_evts].astype(int)
        # r3 = np.zeros([correlated_evts, 7])  # [mod_id, ts, rel_ts, E1, E2, E3, E4]
        r3 = np.zeros(correlated_evts, dtype=self.r3_data_types)  # [mod_id, ts, rel_ts, E1, E2, E3, E4]

        tmp_relative_ts = np.zeros([correlated_evts, 4])  # This exists because I can't broadcast function loads

        for index in np.arange(4):  # This means the order is channel 0, 1, 2, 3 for E1, E2, E3, E4
            channel_events = self.lso_evts[self.module * 4 + index]
            # print('r2:', r2)
            idx_str = 'E' + str(index+1)
            r3[idx_str] = channel_events.col('gate2')[r2] - (3 * channel_events.col('gate2')[r2])
            # r3[:, 3 + index] = channel_events.col('gate2')[r2] - (3 * channel_events.col('gate2')[r2])

            # noinspection PyTypeChecker
            tmp_relative_ts[:, index] = self._time_interp(channel_events, r2)

        # r3[:, 0] = self.module
        r3['mod_id'] = self.module
        # r3[:, 1] = self.scin_evts.col('timestamp')[self.current_bunch_start + r1]
        r3['ts'] = self.lso_evts[self.module * 4].col('timestamp')[r2]
        # r3[:, 2] = np.max(tmp_relative_ts, axis=1) - self.scin_evts.col('timestamp')[self.current_bunch_start + r1]
        r3['rel_ts'] = np.max(tmp_relative_ts, axis=1) - self.scin_evts.col('timestamp')[self.current_bunch_start + r1]
        return r1, r2, r3

    def _time_interp(self, channel_node, gamma_ids):
        ts = (channel_node.col('timestamp')[gamma_ids]).astype(float)  # original timestamps
        # TODO: Check that this astype() is necessary. Copies might slow you down
        maw_max = (channel_node.col('maw_max')[gamma_ids] - 0x8000000).astype(float)
        maw_after = (channel_node.col('maw_after_trig')[gamma_ids] - 0x8000000).astype(float)
        maw_before = (channel_node.col('maw_before_trig')[gamma_ids] - 0x8000000).astype(float)

        time_after = (maw_max/2) - maw_after
        back_interp = maw_before - maw_after

        return ts - (time_after/back_interp)  # This is still in samples


def load_data(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def load_coordinates(calib_file):
    f = open(calib_file, 'r')
    coordinates = eval(f.read().replace(' ', ','))
    f.close()
    return np.array(coordinates)


def pixel_mapper(crds):
    roots = crds[:, :2]  # need to flip, John X is along skyward axis, y along beam. Opposite to mine
    roots[:, [0, 1]] = roots[:, [1, 0]]
    roots[:, 1] *= -1
    roots[:, 1] += 100
    # print('Data points for Tree:, ', spatial.cKDTree(roots).n)
    return spatial.cKDTree(roots)


def main():
    import time
    file = '/Users/justinellin/repos/python_SIS3316/Data/2020-10-08-0958.h5'
    # file = '/Users/justinellin/Desktop/Davis_2020/Thursday/2020-10-08-1119.h5'
    # Position 12, original position (started at 5:30 am)

    # file = '/Users/justinellin/Desktop/Davis_2020/Thursday/2020-10-08-1546.h5'
    # Position 0, edge of table  (started 3:49 am, Done)

    # evt_time_recon = time_recon(file, test=100000, span=48)
    start = time.time()
    evt_time_recon = time_recon(file, span=48)
    print('Good so far')
    evt_time_recon.time_correlate()
    evt_time_recon.h5file.close()
    end = time.time()

    print('That took {s} seconds'.format(s=end-start))


def main2():
    file = '/Users/justinellin/repos/python_SIS3316/processing/processedW-48+48_10-08-1546.h5'
    table = load_data(file)
    bids = table.root.event_data.col('bid')
    multiplicity = np.bincount(np.bincount(bids[:]))
    print('Multiplicity array 1:', multiplicity)

    file2 = '/Users/justinellin/repos/python_SIS3316/processing/processedW-48+48_10-08-1119.h5'
    table2 = load_data(file2)
    bids2 = table2.root.event_data.col('bid')
    multiplicity2 = np.bincount(np.bincount(bids2[:]))
    print('Multiplicity array 2:', multiplicity2)

    m1 = np.zeros(6)
    m1[:multiplicity.size -1] = multiplicity[1:]

    m2 = np.zeros(6)
    m2[:multiplicity2.size -1] = multiplicity2[1:]

    x = np.arange(6) + 1

    plt.step(x, m2, where='mid', label='Position 12, Nearest')
    plt.step(x, m1, where='mid', label='Position 0, Table Edge')
        # plt.step(np.arange(samples), trace, label='mod' + str(ind))
    plt.title('5 Minutes of Data from Two Furthest Points')
    plt.xlabel('Multiplicity')
    plt.legend(loc='best')
    #plt.ylabel('ADC')
    plt.show()


def main3():
    file = '/Users/justinellin/repos/python_SIS3316/processing/processedW-48+48_10-08-1119.h5'
    tab1 = load_data(file)
    # Note these are reversed from main2()
    file2 = '/Users/justinellin/repos/python_SIS3316/processing/processedW-48+48_10-08-1546.h5'
    tab2 = load_data(file2)

    # rng = np.array([-12, 12])
    subsample = 20
    # rawsamples = 24
    rawsamples = 10
    t_bins = np.linspace(-rawsamples/2, rawsamples/2, (rawsamples * subsample) + 1) * 4

    ts1 = tab1.root.event_data.col('rel_ts')
    ts2 = tab2.root.event_data.col('rel_ts')
    # hist1 = np.histogram(ts1[:], bins=t_bins)[0]
    # plt.hist(ts1[:], bins=t_bins, histtype='step', label='Position 0, Table Edge')
    # plt.hist(ts2[:], bins=t_bins, histtype='step', label='Position 12, Nearest')
    plt.hist((4 * ts1[:], 4 * ts2[:]), bins = t_bins, histtype='step', label=['Position 12, Nearest', 'Position 0, Table Edge'])

    plt.title('5 Minutes of Data from Two Furthest Points, Pre-Trigger Delay 8 ns')
    plt.xlabel('Time Relative to Bunch (ns)')
    plt.legend(loc='lower right')
    # x= np.arange
    # plt.step(np.arange(hist1.size), hist1)

    plt.show()


def main4():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evens', '-e', action='store_true', help='read even files')
    args = parser.parse_args()

    choose_evens = args.evens

    f1 = 'str'



if __name__ == "__main__":
    import argparse
    main()
    # main2()
    # main3()
    # main4()
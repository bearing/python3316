import tables
import io
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

# '/Users/justinellin/Desktop/Davis 2020/Tuesday/2020-10-06-1503.h5'
# -> beam axis is +x-hat, away from earth is +y-hat


class h5processed(object):
    def __init__(self, filepath, window):

        raw_samples = 23  # TODO: Check dynamically

        w1str = str(window[0])
        w2str = str(window[1])

        if window[0] >= 0:
            w1str = '+' + w1str

        if window[1] >= 0:
            w2str = '+' + w2str

        # TODO: Check if str?
        self.new_file_name = 'processedW' + w1str + w2str + filepath.replace('.h5', '')[-10:]
        # Should be just the date str
        self.lso_data_fields = ['bid', 'rel_ts', 'E1', 'E2', 'E3', 'E4']
        self.lso_data_types = [np.uint32, np.float32, np.uint32, np.uint32, np.uint32, np.uint32]
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

        # Defining Window

        if isinstance(span, int):
            window = np.array([-span, span])
        else:
            window = np.array(span).sort()

        if window.size > 2 or window.size < 1:
            raise ValueError('Window should be 1 value if symmetric or 2. '
                             'Size {s} provided'.format(s=np.array(window).size))
        self.window = window
        #

        self.processed_file = h5processed(filepath, self.window)

        self.lso_evts = [None] * 64
        for integer in np.arange(64):
            folder = '/det' + str(int(integer))
            self.lso_evts[integer] =  self.h5file.get_node('/', folder).EventData

        scin_folder = '/det' + str(64)
        self.scin_evts = self.h5file.get_node('/', scin_folder).EventData

        if test:  # i.e. test is non-zero
            self.proton_events = test
        else:
            self.proton_events = self.scin_evts.nrows

        self.lso_data_fields = ['bid', 'rel_ts', 'E1', 'E2', 'E3', 'E4']
        self.lso_data_types = [np.uint32, np.float32, np.uint32, np.uint32, np.uint32, np.uint32]
        # scintillator fields: scin_raw, scin_ts
        self.chunk_size = 20000  # number of proton bunches at 1 time
        self.scan_size = 2000  # Number of LSO events to scan at 1 time
        # self.histogram_bins = np.linspace(0, 100000, 3000)

    def time_correlate(self):
        value = self.window
        if isinstance(value, int):
            window = np.array([-value, value])
        else:
            window = np.array(value).sort()

        if window.size > 2 or window.size < 1:
            raise ValueError('Window should be 1 value if symmetric or 2. '
                             'Size {s} provided'.format(s=np.array(window).size))

        # lso_channels =  np.arange(64)  # starts at zero
        # lso_evts = [None] * 64
        mod_idx = np.arange(64//4)  # 16
        mod_store = [None] * 16
        mod_channels = np.arange(4)  # 4 channels per module
        ch_store = [None] * 4
        # scin_channel = 64  # channel 65

        # scin_folder = '/det' + str(scin_channel)
        # self.scin_evts = self.h5file.get_node('/', scin_folder).EventData
        scin_timestamps = self.scin_evts.col('timestamp')

        # if test:
        #    self.proton_events = self.scin_evts.nrows
        # else:
        #    self.proton_events = test

        mod_ts = [None] * 4

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
            if last_evt < self.proton_events:
                current_protons = scin_timestamps[start:last_evt]
                blk_ind += 1
            else:
                current_protons = scin_timestamps[start:]
                process = False

            ref = spatial.cKDTree(current_protons)  # bunches

            for mod_id in mod_idx:
                self.module = mod_id
                mod_ts = self.lso_evts[mod_id * 4].col('timestamp')
                # for integer in mod_channels:
                #    folder = '/det' + str(int(4 * mod_id + integer))
                #    mod_ts[integer] = self.h5file.get_node('/', folder).EventData.col('timestamp')
                self._time_correlate_module(ref, mod_ts, window)  # Do I need mod id?

            process = False  # TODO: Delete this when ready

    def _time_correlate_module(self, scin_ts_tree, mod_ts, window):
        pass


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
    file = '/Users/justinellin/repos/python_SIS3316/Data/2020-10-08-0958.h5'


if __name__ == "__main__":
    main()
    # main2()
    # main3()

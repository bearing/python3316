import numpy as np
import collections
import tables
from common import hardware_constants

# from timeit import default_timer as timer


class h5f(object):

    # data_fields = ['format', 'det', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'pileup',
    # 'repileup','gate2', 'gate3', 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig',
    # 'maw_before_trig', 'en_start', 'en_max', 'raw_data', 'maw_data']

    _options = ('raw', 'parsed')

    def __init__(self, save_fname, hit_stats, data_save_type='parsed'):
        if data_save_type not in self._options:
            raise ValueError('Save type {df} is not supported. '
                             'Supported file types: {opt}'.format(df=data_save_type, opt=str(self._options))[1:-1])
        self.file = tables.open_file(save_fname, mode="w", title="Acquisition Data File")

        if data_save_type == 'raw':
            self._h5_raw(self.file, hit_stats)

        if data_save_type == 'parsed':
            self._h5_parsed(self.file, hit_stats)

    def __del__(self):
        """ Run this manually if you need to close file."""
        self.file.close()

    def _h5_raw(self, file, hit_fmts):  # hit_fmts is hit_stats
        """ Sets up file structure for hdf5 """

        det = 65  # hardcoded, 0-63 is LSO, 64 is scintillator

        for ind in np.arange(det):
            template = hit_fmts[ind]
            if ind is not 64:
                grp = file.create_group("/", 'det' + str(ind), 'Det' + str(ind) + 'Data')
            else:
                grp = file.create_group("/", 'det' + str(ind), 'Plastic Data')

            raw_event_length = template['event length']//2

            self.file.create_earray(grp, 'FPGA words', atom=tables.atom.UInt32Atom(), shape=(0, raw_event_length))

        self.file.flush()

    def _h5_parsed(self, file, hit_fmts):  # hit_fmts is hit_stats
        """ Sets up file structure for hdf5 """

        # data_fields = ['format', 'det', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'gate2',
        # 'gate7', 'gate8', 'maw_max', 'maw_after_trig', 'maw_before_trig', 'en_max',
        #                'en_start', 'raw_data', 'maw_data']

        # det = len(hit_fmts)
        det = 65  # hardcoded, 0-63 is LSO, 64 is scintillator

        for ind in np.arange(det):  # FIXME: CHANNEL 0 MUST BE PLASTIC SCINTILLATOR ON 5th MODULE
            template = hit_fmts[ind]
            if ind is not 64:
                grp = file.create_group("/", 'det' + str(ind), 'Det' + str(ind) + 'Data')
            else:
                grp = file.create_group("/", 'det' + str(ind), 'Plastic Data')

            hit_fields = ['rid', 'timestamp']
            data_types = [np.uint32, np.uint64]

            if bool(template['acc1_flag']):
                hit_fields.extend(['adc_max', 'adc_argmax', 'pileup', 'gate1', 'gate2'])
                data_types.extend([np.uint16, np.uint16, np.uint8, np.uint32, np.uint32])

                # hit_fields.extend(['adc_max', 'adc_argmax', 'pileup', 'gate1', 'gate2', 'gate3', 'gate4', 'gate5',
                #                   'gate6'])
                # data_types.extend([np.uint16, np.uint16, np.uint8, np.uint32, np.uint32, np.uint32, np.uint32,
                #                   np.uint32, np.uint32])

            if bool(template['acc2_flag']):
                hit_fields.extend(['gate7', 'gate8'])
                data_types.extend([np.uint32, np.uint32])

            if bool(template['maw_flag']):
                hit_fields.extend(['maw_max', 'maw_before_trig', 'maw_after_trig'])
                data_types.extend([np.uint32, np.uint32, np.uint32])

            if bool(template['maw_energy_flag']):
                hit_fields.extend(['en_start', 'en_max'])
                data_types.extend([np.uint32, np.uint32])

            sis3316_dtypes = np.dtype({"names": hit_fields, "formats": data_types})

            # edata_table = self.file.create_table('/', 'EventData', description=sis3316_dtypes)
            self.file.create_table(grp, 'EventData', description=sis3316_dtypes)

            raw_samples = template['raw_event_length']
            if raw_samples:
                # Create RawData array
                self.file.create_earray(grp, 'raw_data', atom=tables.atom.UInt16Atom(), shape=(0, raw_samples))

            maw_samples = template['maw_event_length']
            if maw_samples:
                # Create MAWData array
                self.file.create_earray(grp, 'maw_data', atom=tables.atom.UInt32Atom(), shape=(0, maw_samples))

        self.file.flush()

    def save(self, data_dict, evts, *args):
        if not evts:  # i.e. check for no events and then skip the rest
            return
        try:
            # print("Events:", evts)
            base_node = '/det'
            detid = _det_from_args(*args)
            # if detid > 64:  # hardcoded
            #    detid = 64
            base_node += str(detid)

            for table in self.file.iter_nodes(base_node, classname='Table'):  # Structured data sets
                data_struct = np.zeros(evts, dtype=table.description._v_dtype)
                for field in table.description._v_names:  # Field functions as a key
                    if data_dict[field] is not None:
                        data_struct[field] = data_dict[field]
                table.append(data_struct)
                table.flush()

            for earray in self.file.iter_nodes(base_node, classname='EArray'):  # Homogeneous data sets
                earray.append(data_dict[earray.name])
                earray.flush()

            self.file.flush()

        except Exception as e:
            print(e)


def _det_from_args(*args):
    if len(args) is 2:
        board = np.array(args[0])
        channel = np.array(args[1])
        det = hardware_constants.CHAN_TOTAL * board + channel
    else:
        det = args
    return det
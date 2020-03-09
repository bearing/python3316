import numpy as np
import collections
import tables
from common import hardware_constants

# from timeit import default_timer as timer


class h5f(object):
    # TODO: Allow channels to be skipped entirely if none of their trigger settings are enabled i.e. channel.flags or
    #  instead set the event, raw, maw lengths to zero and skip those

    # data_fields = ['format', 'det', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'pileup',
    # 'repileup','gate2', 'gate3', 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig',
    # 'maw_before_trig', 'en_start', 'en_max', 'raw_data', 'maw_data']

    _options = ('raw_hdf5', 'recon_hdf5')

    def __init__(self, save_fname, hit_stats, data_save_type='raw_hdf5'):
        if data_save_type not in self._options:
            raise ValueError('Save type {df} is not supported. '
                             'Supported file types: {opt}'.format(df=data_save_type, opt=str(self._options))[1:-1])
        self.file = tables.open_file(save_fname, mode="w", title="Acquisition Data File")
        # self.hit_stats = hit_stats
        # self.same_settings = self._check_events(hit_stats)  # True means that every channel is saving with same format

        if data_save_type == 'raw_hdf5':
            self.same_settings = self._check_events(hit_stats)
            self._h5_raw_file_setup(self.file, hit_stats, same=self.same_settings)

        if data_save_type == 'recon_hdf5':
            self.same_settings = True  # One layer deep
            self._h5_recon_file_setup()
            # self._h5_recon_file_setup(self.file, hit_stats, same=self.same_settings)

    def __del__(self):
        """ Run this manually if you need to close file."""
        self.file.close()

    def _check_events(self, hit_stats):  # This is being done explicitly in case someone else needs to modify this
        num_det = len(hit_stats)
        evt_lengths = [None] * num_det
        raw_samples = [None] * num_det
        maw_samples = [None] * num_det

        for det, stats in enumerate(hit_stats):
            evt_lengths[det] = stats['event_length']
            raw_samples[det] = stats['raw_event_length']
            maw_samples[det] = stats['maw_event_length']

        return (evt_lengths.count(evt_lengths[0]) == num_det) & (raw_samples.count(raw_samples[0]) == num_det) &\
               (maw_samples.count(maw_samples[0]) == num_det)

    def _h5_raw_file_setup(self, file, hit_fmts, same=True):  # hit_fmts is hit_stats
        """ Sets up file structure for hdf5 """

        # data_fields = ['format', 'det', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'gate2', 'gate3',
        # 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig', 'maw_before_trig', 'en_max',
        #                'en_start', 'raw_data', 'maw_data']

        if same:
            hit_fields = ['rid', 'det', 'timestamp']
            data_types = [np.uint32, np.uint8, np.uint64]

            template = hit_fmts[0]  # since they are all the same, just use the first one

            if bool(template['acc1_flag']):
                hit_fields.extend(['adc_max', 'adc_argmax', 'pileup', 'gate1', 'gate2', 'gate3', 'gate4', 'gate5',
                                   'gate6'])
                data_types.extend([np.uint16, np.uint16, np.uint8, np.uint32, np.uint32, np.uint32, np.uint32,
                                   np.uint32, np.uint32])

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
            self.file.create_table('/', 'event_data', description=sis3316_dtypes)

            raw_samples = template['raw_event_length']
            if raw_samples:
                # Create RawData array
                self.file.create_earray('/', 'raw_data', atom=tables.atom.UInt16Atom(), shape=(0, raw_samples))

            maw_samples = template['maw_event_length']
            if maw_samples:
                # Create MAWData array
                self.file.create_earray('/', 'maw_data', atom=tables.atom.UInt32Atom(), shape=(0, maw_samples))

            # self.file.flush()
        else:
            Warning("All channels do not have the same event formats set. Saving to individual folders!")

            det = len(hit_fmts)

            for ind in np.arange(det):
                template = hit_fmts[ind]
                grp = file.create_group("/", 'det' + str(ind), 'Det' + str(ind) + 'Data')

                hit_fields = ['rid', 'det', 'timestamp']
                data_types = [np.uint32, np.uint8, np.uint64]

                if bool(template['acc1_flag']):
                    hit_fields.extend(['adc_max', 'adc_argmax', 'pileup', 'gate1', 'gate2', 'gate3', 'gate4', 'gate5',
                                       'gate6'])
                    data_types.extend([np.uint16, np.uint16, np.uint8, np.uint32, np.uint32, np.uint32, np.uint32,
                                       np.uint32, np.uint32])

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

    # def _h5_recon_file_setup(self, file, hit_fmts, same=True):
    def _h5_recon_file_setup(self):

        recon_fields = ['energy', 'crystal_index', 'timestamp']
        data_types = [np.float64, np.uint16, np.uint64]
        LSO_dtypes = np.dtype({"names": recon_fields, "formats": data_types})
        self.file.create_table('/', 'recon_data', description=LSO_dtypes)

    def save(self, data_dict, evts, *args):
        if not evts:  # i.e. check for no events and then skip the rest
            return
        try:
            print("Events:", evts)
            base_node = '/'
            if not self.same_settings:  # More than one layer deep
                base_node += 'det' + str(_det_from_args(*args))

            # TODO: Check this works. If not have to check for each class type

            for table in self.file.iter_nodes(base_node, classname='Table'):  # Structured data sets
                print("Table description:", table.description._v_dtype)
                data_struct = np.zeros(evts, dtype=table.description._v_dtype)
                for field in table.description._v_names:  # Field functions as a key
                    if data_dict[field] is not None:
                        print("Field:", field)
                        print("Data Dict[field]:", data_dict[field])
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

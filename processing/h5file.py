import numpy as np
import collections
import tables

# from timeit import default_timer as timer


class h5f(object):
    # TODO: Allow channels to be skipped entirely if none of their trigger settings are enabled i.e. channel.flags or
    #  instead set the event, raw, maw lengths to zero and skip those

    # data_fields = ['format', 'det', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'pileup',
    # 'repileup','gate2', 'gate3', 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig',
    # 'maw_before_trig', 'en_start', 'en_max', 'raw_data', 'maw_data']

    _options = ('raw', 'recon')

    def __init__(self, save_fname, hit_stats, data_save_type='raw'):
        if data_save_type not in self._options:
            raise ValueError('Save type {df} is not supported. '
                             'Supported file types: {opt}'.format(df=data_save_type, opt=str(self._options))[1:-1])
        self.file = tables.open_file(save_fname, mode="w", title="Acquisition Data File")
        # self.hit_stats = hit_stats
        self.same_settings = self._check_events(hit_stats)  # True means that every channel is saving with same format

        if data_save_type is 'raw':
            self._h5_raw_file_setup(self.file, hit_stats, same=self.same_settings)

        if data_save_type is 'recon':
            self._h5_recon_file_setup(self.file, hit_stats, same=self.same_settings)

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
            hit_fields = ['det', 'timestamp']
            data_types = [np.uint8, np.uint64]

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
            self.file.create_table('/', 'EventData', description=sis3316_dtypes)

            raw_samples = template['raw_event_length']
            if raw_samples:
                # Create RawData array
                self.file.create_earray('/', 'RawData', atom=tables.atom.UInt16Atom(), shape=(0, raw_samples))

            maw_samples = template['maw_event_length']
            if maw_samples:
                # Create MAWData array
                self.file.create_earray('/', 'MAWData', atom=tables.atom.UInt32Atom(), shape=(0, raw_samples))
        else:
            print("Not yet supported!")


    def _h5_recon_file_setup(self, file, hit_fmts, same=True):

        max_ch = len(hit_fmts)
        ch_group = [None] * max_ch

        for ind in np.arange(max_ch):
            ch_group[ind] = file.create_group("/", 'det' + str(ind), 'Data')
            # TODO: 2/10/20 Fix the Folder Organization
            if bool(hit_fmts[ind]['acc1_flag']):
                # hit_fields.extend['adc_max', 'adc_argmax', 'gate1', 'gate2', 'gate3', 'gate4', 'gate5', 'gate6']
                # data_types.extend[]
                pass  # Set up first accumulator flag data types
            if bool(hit_fmts[ind]['acc2_flag']):
                pass  # Set up second accumulator flag data types
            if bool(hit_fmts[ind]['maw_flag']):
                pass  # Set up data types for maw trigger values
            if bool(hit_fmts[ind]['maw_energy_flag']):
                pass  # set up data types for FIR Maw (energy) values
            if hit_fmts[ind]['raw_event_length'] > 0:  # These lengths are defined to 16 bit words (see channel.py)
                pass  # Add Raw Data Group
            if hit_fmts[ind]['maw_event_length'] > 0:
                pass  # Save MAW Data
            pass
        return True
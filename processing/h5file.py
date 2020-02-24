import numpy as np
import collections
import tables

# from timeit import default_timer as timer


class h5f(object):
    """This class performs event reconstruction for the LSO. With detector face normal, the channels are 0 -> 3
    clockwise starting from upper left (so 0 shares y with 1, and shares x with 3) """
    # TODO: Allow channels to be skipped entirely if none of their trigger settings are enabled i.e. channel.flags or
    #  instead set the event, raw, maw lengths to zero and skip those

    # data_fields = ['format', 'det', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'pileup',
    # 'repileup','gate2', 'gate3', 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig',
    # 'maw_before_trig', 'en_start', 'en_max', 'raw_data', 'maw_data']

    def __init__(self, save_fname, hit_stats, raw=True):
        self.file = tables.open_file(save_fname, mode="w", title="Acquisition Data File")
        self.hit_stats = hit_stats
        self.same_settings = self._check_events(hit_stats)  # True means that every channel is saving the same data
        self._h5_file_setup(self.file, hit_stats, self.same_settings)

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

    def _h5_file_setup(self, file, hit_fmts, same=True):
        """ Sets up file structure for hdf5 """

        # data_fields = ['format', 'det', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'gate2', 'gate3',
        # 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig', 'maw_before_trig', 'en_max',
        #                'en_start', 'raw_data', 'maw_data']

        hit_fields = ['det', 'timestamp']
        data_types = [np.uint8, np.uint64]

        max_ch = len(hit_fmts) * 4
        ch_group = [None] * max_ch
        # TODO: ADD HDF5  Datatype Support (1/4/2020)
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
            if bool(hit_fmts[ind]['maw_max_values']):
                pass  # set up data types for FIR Maw (energy) values
            if hit_fmts[ind]['raw_event_length'] > 0:  # These lengths are defined to 16 bit words (see channel.py)
                pass  # Add Raw Data Group
            if hit_fmts[ind]['maw_event_length'] > 0:
                pass  # Save MAW Data
            pass

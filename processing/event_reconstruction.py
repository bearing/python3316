import numpy as np
import collections
from common import hardware_constants
# from timeit import default_timer as timer


class event_histories(object):
    """This class performs event reconstruction for the LSO. With detector face normal, the channels are 0 -> 3
    clockwise starting from upper left (so 0 shares y with 1, and shares x with 3) """
    # -> x is beam direction. Upper left most pixel is (0,0)

    # data_fields = ['format', 'det', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'pileup',
    # 'repileup','gate2', 'gate3', 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig',
    # 'maw_before_trig', 'en_start', 'en_max', 'raw_data', 'maw_data']

    def __init__(self, crystal_file, energy_calibration_file):
        self.crystal_indices = self._load_crystal_indices(crystal_file)
        self.energy_calibration = self._load_energy_calib(energy_calibration_file)
        self.energies = None
        self.timestamps = None
        # self.pixels = None  # Histogrammed center of mass (Anger Logic)

    def _load_crystal_indices(self, crystal_file):
        return True

    def _load_energy_calib(self, energy_file):
        return True

    def process_hits(self, hits):  # should be the data dict
        if not isinstance(hits, collections.Mapping):
            TypeError("Dict, OrderedDict, or UserDict required. {ty} provided.".format(ty=type(hits)))

        det = hits['det'][0]
        cid = det & 0b11

        if cid:
            self.energies[:, cid] = hits['gate2'] - 3 * hits['gate1']
            self.timestamps[:, cid] = hits['timestamp']

        else:  # cid = 0
            num_events = hits['det'].size
            self.energies = np.zeros([num_events, 4])
            self.timestamps = np.zeros([num_events, 4])
            # self.pixels = np.zeros([num_events, 4])

            self.energies[:, 0] = hits['gate2'] - 3 * hits['gate1']
            self.timestamps[:, 0] = hits['timestamp']

        if cid is 3:

            gid = (det >> 2) & 0b11
            board = (det >> 8) - 1

            ul = 0  # upper left
            ur = 1
            ll = 3  # lower left (clockwise)
            lr = 2  # lower right

            # TODO: CHECK YOUR CABLING!
            sum_ind = (np.min(self.timestamps, axis=1) == np.max(self.timestamps, axis=1))
            total_energy = np.sum(self.energies[sum_ind, :], axis=1)
            com_x = (self.energies[sum_ind, ur] + self.energies[sum_ind, lr]) - \
                    (self.energies[sum_ind, ul] + self.energies[sum_ind, ll]) / total_energy

            crystal_x = np.searchsorted(self.crystal_indices['x_bins'], com_x, side='left')

            com_y = (self.energies[sum_ind, ul] + self.energies[sum_ind, ur]) - \
                    (self.energies[sum_ind, ll] + self.energies[sum_ind, lr]) / total_energy

            crystal_y = np.searchsorted(self.crystal_indices['y_bins'], com_y, side='left')

            # TODO: Check this!

            crystal_ind = crystal_x + 12 * crystal_y + (144 * ((4 * board) + gid))  # 12 4 mm pixels per row per PMT

            return {'energy': total_energy, 'crystal_index': crystal_ind, 'timestamps': self.timestamps[sum_ind],
                    'histogram': np.bincount(crystal_ind), 'events': sum_ind.size}

            # 'LSO_module': (4 * board) + gid

        # crystal_ind assumes that, if facing the detectors from the collimator that the first detector is in the
        # upper left then goes from upper left to upper right. The 5 detector (4 with 0 base numbering) is the one below
        # the upper left, etc. Within a detector the numbering is also from upper left to lower right.


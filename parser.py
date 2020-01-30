import numpy as np
from common import hardware_constants
# from timeit import default_timer as timer


class parser(object):
    """This class performs on the fly parsing of data from SIS3316 card(s) based on set config settings"""
    def __init__(self, boards):
        self.boards = boards
        self.event_data = [c.event_stats for b in self.boards for c in b._chan]

    def update(self, board_numbers, chan_numbers):
        """Allows for updating expected hit/event data after configuration is first set."""
        # TODO: Add board_id to module_manager.py to make this easier
        try:
            b = np.array(board_numbers)
            c = np.array(chan_numbers)
            up_ind = hardware_constants.CHAN_TOTAL * b + c
            for idx in np.arange(len(up_ind)):
                self.event_data[up_ind[idx]] = self.boards[b[idx]]._chan[chan_numbers[idx]].event_stats
        except Exception as e:
            print(e)

    def parse(self, buffer, channel):
        """On the fly parser. Needs a buffer object of data and the index of the channel. Returns a dictionary """
        # TODO: 1/30/19 Read readout_buffer

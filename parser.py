import numpy as np
from common import hardware_constants
# from timeit import default_timer as timer


class parser(object):
    """This class performs on the fly parsing of data from SIS3316 card(s) based on set config settings"""

    def __init__(self, boards):
        self.boards = boards
        self.event_data = [c.event_stats for b in self.boards for c in b._chan]

    wordtype = np.dtype(np.uint16).newbyteorder('<')  # Events are read as 16 bit words, small endian

    def update(self, *args):
        """Allows for updating expected hit/event data after configuration is first set. Currently not used"""
        # TODO: Add board_id to module_manager.py to make this easier
        try:
            if len(args) is 2:
                board = np.array(args[0])
                channel = np.array(args[1])
                up_ind = hardware_constants.CHAN_TOTAL * board + channel
            else:
                up_ind = np.array(args)
                board = (up_ind // hardware_constants.CHAN_TOTAL).astype(int)
                channel = up_ind % hardware_constants.CHAN_TOTAL
            for idx in np.arange(len(up_ind)):
                self.event_data[up_ind[idx]] = self.boards[board[idx]]._chan[channel[idx]].event_stats
        except Exception as e:
            print(e)

    def parse(self, buffer, *args):
        """On the fly parser. Needs a buffer object of data and the index of the channel. Returns a dictionary """
        if len(args) is 2:
            channel = hardware_constants.CHAN_TOTAL * args[0] + args[1]  # module and channel number
        else:
            channel = args  # system channel number
        current_event = self.event_data[channel]
        event_length = current_event['event_length']
        raw = np.frombuffer(buffer, dtype=self.wordtype)
        if raw.size is 0 or int(event_length) is 0:
            return

        try:
            event_arr = raw.reshape([(raw.size//event_length), event_length])
            # array where each row is an event


            data = {}

            # Always get these
            ch_fmt = event_arr[:, 0]
            fmt = ch_fmt & 0b111
            ch = ch_fmt >> 4
            data['timestamp'] = (event_arr[:, 1] << 32) + (event_arr[:, 2]) + (event_arr[:, 3] << 16)

            pos = 4  # pointer to column where you are reading

            if bool(current_event['acc1_flag']):
                data['adc_max'] = event_arr[:, 4]
                data['adc_argmax'] = event_arr[:, 5]
                data['gate1'] = event_arr[:, 6] + ((event_arr[:, 7] & 0xFF) << 16)
                data['gate2'] = event_arr[8] + (event_arr[9] << 16)
                data['gate3'] = event_arr[10] + (event_arr[11] << 16)
                data['gate4'] = event_arr[12] + (event_arr[13] << 16)
                data['gate5'] = event_arr[14] + (event_arr[15] << 16)
                data['gate6'] = event_arr[16] + (event_arr[17] << 16)

            pos += 14

        except Exception as e:
            # TODO: write to raw file and spit out error file
            print(e)

        if bool(current_event['acc2_flag']):
            pass  # Set up second accumulator flag data types
        if bool(current_event['maw_flag']):
            pass  # Set up data types for maw trigger values
        if bool(current_event['maw_max_values']):
            pass  # set up data types for FIR Maw (energy) values
        if current_event['maw_event_length'] > 0:
            pass  # Save MAW Data
        pass
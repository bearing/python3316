import numpy as np
from common import hardware_constants
# from timeit import default_timer as timer


class parser(object):
    """This class performs on the fly parsing of data from SIS3316 card(s) based on set config settings"""

    _options = ('raw', 'parsed')  # Davis

    def __init__(self, boards, data_save_type='parsed'):
        if data_save_type not in self._options:
            raise ValueError('Save type {df} is not supported. '
                             'Supported file types: {opt}'.format(df=data_save_type, opt=str(self._options))[1:-1])
        self.parse_option = data_save_type  # Davis
        self.boards = boards
        self.event_data = [c.event_stats for b in self.boards for c in b._chan]
        self.event_id = 1  # Starts from 1

    ADCWORDSIZE = np.dtype(np.uint16).newbyteorder('<')  # Events are read as 16 bit words, small endian
    FPGAWORDSIZE = np.dtype(np.uint32).newbyteorder('<')  # Events are read as 32 bit words, small endian

    def update(self, *args):
        """Allows for updating expected hit/event data after configuration is first set. Currently not used"""
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

    def parse(self, *args):  # Davis: Bizarrely clumsy
        if self.parse_option == 'parsed':
            data, evts = self.parse_fields(*args)
        else:
            data, evts = self.parse_raw(*args)
        return data, evts

    def parse_fields(self, buffer, *args):
        """On the fly parser. Needs a buffer object of data and the index of the channel. Returns a dictionary """
        # 32 bit words
        if len(args) is 2:
            detector = hardware_constants.CHAN_TOTAL * args[0] + args[1]  # module and channel number
        else:
            detector = args[0]  # detector number
        # module = detector // hardware_constants.CHAN_GRP_COUNT

        current_event = self.event_data[detector]
        event_length = current_event['event_length'] // 2

        if buffer.size is 0 or int(event_length) is 0:
            return None, None

        try:
            data = {}

            evts = buffer.size // event_length
            # event_arr = new_buffer.reshape([evts, event_length])  # rows are events
            event_arr = buffer.reshape([evts, event_length]).T  # columns are events

            data['rid'] = self.event_id + np.arange(evts)
            self.event_id += evts

            # Always get these
            # ch_fmt = event_arr[:, 0] & 0xFFFF  # If events are as rows
            ch_fmt = event_arr[0] & 0xFFFF

            fmt = ch_fmt & 0b1111

            # if np.max(fmt) is not np.min(fmt):
            #    pass  # TODO: ERROR. save to raw instead
            # data['format'] = fmt

            # data['det'] = ch_fmt >> 4  # Davis, redundant information
            # data['channel'] = (ch & 0b1111)  # Hardwired
            # data['module'] = (ch >> 4)  # Writeable. Can be module number allowing for detector number

            data['timestamp'] = ((event_arr[0].astype(np.uint64) & 0xFFFF0000) << 16) + event_arr[1]

            pos = 2  # pointer to column where you are reading

            if bool(current_event['acc1_flag']):
                data['adc_max'] = event_arr[pos] & 0xFFFF
                data['adc_argmax'] = event_arr[pos] >> 16
                data['gate1'] = event_arr[pos+1] & 0xFFFFff  # 24 bits

                info = (event_arr[pos + 1] >> (24 + 4))  # First 4 bits "reserved"
                data['pileup'] = info & 0b11  # This should really be triggers. 1 is normal.
                # data['underflow'] = info & 0b100
                # data['overflow'] = info & 0b1000

                data['gate2'] = event_arr[pos + 2]
                # data['gate3'] = event_arr[pos + 3] # Davis = Not used
                # data['gate4'] = event_arr[pos + 4] # Davis = Not used
                # data['gate5'] = event_arr[pos + 5] # Davis = Not used
                # data['gate6'] = event_arr[pos + 6] # Davis = Not used
                pos += 7

            if bool(current_event['acc2_flag']):
                data['gate7'] = event_arr[pos]
                data['gate8'] = event_arr[pos + 1]
                pos += 2

            if bool(current_event['maw_flag']):
                data['maw_max'] = event_arr[pos]
                data['maw_before_trig'] = event_arr[pos + 1]
                data['maw_after_trig'] = event_arr[pos + 2]
                pos += 3

            if bool(current_event['maw_energy_flag']):
                data['en_start'] = event_arr[pos]
                data['en_max'] = event_arr[pos + 1]
                pos += 2

            raw_samples = event_arr[pos] & 0x3FFffff  # in 32-bit words
            OxE_maw_status = event_arr[pos] >> 26
            OxE, maw_test, status_flag = OxE_maw_status >> 2, (OxE_maw_status >> 1 & 0b1), (OxE_maw_status & 0b1)
            pos += 1

            if np.min(OxE) is not np.max(OxE):  # Fast way to check all values are equal
                pass  # TODO: ERROR. save to raw instead

            if (raw_samples * 2) is not current_event['raw_event_length']:
                ValueError("Buffer wants to return {r} samples but channel is "
                           "set to {b} samples!".format(r=raw_samples*2, b=current_event['raw_event_length']))

            if np.min(raw_samples) is not np.max(raw_samples):
                ValueError("Something is wrong. Raw samples return more than 1 value: {v}".format(v=raw_samples))

            raw_samples = 2 * raw_samples[0]  # in 16 bit words

            if raw_samples:
                raw_words = event_arr[pos:(pos + raw_samples)]

                raw_data = np.zeros([raw_samples, evts], dtype=np.uint16)
                raw_data[0::2] = raw_words & 0xFFFF
                raw_data[1::2] = raw_words >> 16

                data['raw_data'] = raw_data.T  # Transpose this into rows
                pos += raw_samples

            maw_samples = current_event['maw_event_length']
            if maw_samples:
                data['maw_data'] = event_arr[pos:(pos + maw_samples)].T

            return data, evts

        except Exception as e:
            # TODO: write to raw file and spit out error file
            print(e)

    def parse_raw(self, buffer, *args):
        """On the fly parser. Needs a buffer object of data and the index of the channel. Returns a dictionary """
        # 32 bit words
        if len(args) is 2:
            detector = hardware_constants.CHAN_TOTAL * args[0] + args[1]  # module and channel number
        else:
            detector = args[0]  # detector number
        # module = detector // hardware_constants.CHAN_GRP_COUNT

        current_event = self.event_data[detector]
        event_length = current_event['event_length'] // 2

        if buffer.size is 0 or int(event_length) is 0:
            return None, None

        try:
            data = {}

            evts = buffer.size // event_length
            data['raw'] = buffer.reshape([evts, event_length])  # rows are events
            return data, evts

        except Exception as e:
            print(e)


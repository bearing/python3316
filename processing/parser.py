import numpy as np
from common import hardware_constants
# from timeit import default_timer as timer


class parser(object):
    """This class performs on the fly parsing of data from SIS3316 card(s) based on set config settings"""

    def __init__(self, boards):
        self.boards = boards
        self.event_data = [c.event_stats for b in self.boards for c in b._chan]
        self.event_id = 1  # Starts from 1

    ADCWORDSIZE = np.dtype(np.uint16).newbyteorder('<')  # Events are read as 16 bit words, small endian
    FPGAWORDSIZE = np.dtype(np.uint32).newbyteorder('<')  # Events are read as 32 bit words, smallendian

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
        # 32 bit words
        if len(args) is 2:
            detector = hardware_constants.CHAN_TOTAL * args[0] + args[1]  # module and channel number
        else:
            detector = args  # detector number
        current_event = self.event_data[detector]
        event_length = current_event['event_length'] // 2

        # first_event_buffer_mode = current_event['save_first_raw_only'] & current_event['raw_event_length']
        first_event_buffer_mode = current_event['save_first_raw_only']
        # This mode saves the raw waveform only for the first event in the bank

        if buffer.size is 0 or int(event_length) is 0:
            return None, None

        try:
            data = {}

            if first_event_buffer_mode:
                # You can run this and swap on single events, but it will be slower
                first_evt = buffer[:event_length]
                rest = buffer[event_length:]
                offset = 2 + bool(current_event['acc1_flag']) * 7 \
                         + bool(current_event['acc2_flag']) * 2 \
                         + bool(current_event['maw_flag']) * 3 \
                         + bool(current_event['maw_energy_flag']) * 2
                raw_samples = 2 * (first_evt[offset] & 0x3FFffff)  # number of 16 bit words
                offset += 1
                end = offset + raw_samples  # of raw
                raw_words = first_evt[offset:end]

                raw_data = np.zeros(raw_samples, dtype=np.uint16)
                raw_data[0::2] = raw_words & 0xFFFF
                raw_data[1::2] = raw_words >> 16
                data['raw_data'] = raw_data

                new_buffer = np.zeros(buffer.size - raw_samples, dtype=buffer.dtype)
                new_buffer[:offset] = buffer[:offset]
                new_buffer[offset:] = buffer[end:]
                # new_buffer = np.concatenate((first_evt[:offset], first_evt[end:], rest))

                event_length -= current_event['raw_event_length'] // 2  # All except first

            else:
                new_buffer = buffer

            evts = new_buffer.size // event_length
            # event_arr = new_buffer.reshape([evts, event_length])  # rows are events
            event_arr = new_buffer.reshape([evts, event_length]).T  # columns are events

            if first_event_buffer_mode:
                rid = np.zeros(evts)
                rid[0] = self.event_id
                data['rid'] = rid  # Clumsy, lot of zeroes but oh well
                self.event_id += 1
            else:
                data['rid'] = self.event_id + np.arange(evts)
                self.event_id += evts

            # data = {'rid': self.event_id + np.arange(evts)}
            # data['rid'] = self.event_id + np.arange(evts)
            # self.event_id += evts

            # Always get these
            # ch_fmt = event_arr[:, 0] & 0xFFFF  # If events are as rows
            ch_fmt = event_arr[0] & 0xFFFF

            fmt = ch_fmt & 0b1111

            if np.max(fmt) is not np.min(fmt):
                pass  # TODO: ERROR. save to raw instead
            # data['format'] = fmt

            data['det'] = ch_fmt >> 4
            # data['channel'] = (ch & 0b1111)  # Hardwired
            # data['module'] = (ch >> 4)  # Writeable. Can be module number allowing for detector number

            # data['timestamp'] = ((event_arr[:, 0].astype(np.uint64) & 0xFFFF0000) << 16) + event_arr[:, 1]
            data['timestamp'] = ((event_arr[0].astype(np.uint64) & 0xFFFF0000) << 16) + event_arr[1]

            pos = 2  # pointer to column where you are reading

            if bool(current_event['acc1_flag']):
                # data['adc_max'] = event_arr[:, pos] & 0xFFFF
                # data['adc_argmax'] = event_arr[:, pos] >> 16
                # data['gate1'] = event_arr[:, (pos+1)] & 0xFFFFff  # 24 bits
                data['adc_max'] = event_arr[pos] & 0xFFFF
                data['adc_argmax'] = event_arr[pos] >> 16
                data['gate1'] = event_arr[pos+1] & 0xFFFFff  # 24 bits

                info = (event_arr[pos + 1] >> (24 + 4))  # First 4 bits "reserved"
                # info = (event_arr[:, (pos+1)] >> (24 + 4))
                data['pileup'] = info & 0b11  # This should really be triggers. 1 is normal.
                # data['underflow'] = info & 0b100
                # data['overflow'] = info & 0b1000

                # data['gate2'] = event_arr[:, pos + 2]
                # data['gate3'] = event_arr[:, pos + 3]
                # data['gate4'] = event_arr[:, pos + 4]
                # data['gate5'] = event_arr[:, pos + 5]
                # data['gate6'] = event_arr[:, pos + 6]
                data['gate2'] = event_arr[pos + 2]
                data['gate3'] = event_arr[pos + 3]
                data['gate4'] = event_arr[pos + 4]
                data['gate5'] = event_arr[pos + 5]
                data['gate6'] = event_arr[pos + 6]
                pos += 7

            if bool(current_event['acc2_flag']):
                # data['gate7'] = event_arr[:, pos]
                # data['gate8'] = event_arr[:, pos+1]
                data['gate7'] = event_arr[pos]
                data['gate8'] = event_arr[pos + 1]
                pos += 2

            if bool(current_event['maw_flag']):
                # data['maw_max'] = event_arr[:, pos]
                # data['maw_before_trig'] = event_arr[:, (pos + 1)]
                # data['maw_after_trig'] = event_arr[:, (pos + 2)]
                data['maw_max'] = event_arr[pos]
                data['maw_before_trig'] = event_arr[pos + 1]
                data['maw_after_trig'] = event_arr[pos + 2]
                pos += 3

            if bool(current_event['maw_energy_flag']):
                # data['en_start'] = event_arr[:, pos]
                # data['en_max'] = event_arr[:, (pos + 1)]
                data['en_start'] = event_arr[pos]
                data['en_max'] = event_arr[pos + 1]
                pos += 2

            # raw_samples = event_arr[:, pos] & 0x3FFffff  # in 32-bit words
            # OxE_maw_status = event_arr[:, pos] >> 26
            raw_samples = event_arr[pos] & 0x3FFffff  # in 32-bit words
            OxE_maw_status = event_arr[pos] >> 26
            OxE, maw_test, status_flag = OxE_maw_status >> 2, (OxE_maw_status >> 1 & 0b1), (OxE_maw_status & 0b1)
            # data['maw_select'] = maw_test  # Trigger MAW (0) or Energy MAW (1)
            # data['status'] = status_flag   # Pileup or Repileup
            pos += 1

            if np.min(OxE) is not np.max(OxE):  # Fast way to check all values are equal
                pass  # TODO: ERROR. save to raw instead

            if (raw_samples * 2) is not current_event['raw_event_length'] and first_event_buffer_mode is False:
                ValueError("Buffer wants to return {r} samples but channel is "
                           "set to {b} samples!".format(r=raw_samples*2, b=current_event['raw_event_length']))

            if np.min(raw_samples) is not np.max(raw_samples):
                ValueError("Something is wrong. Raw samples return more than 1 value: {v}".format(v=raw_samples))

            raw_samples = 2 * raw_samples[0]  # in 16 bit words

            if raw_samples and not first_event_buffer_mode:
                # raw_words = event_arr[:, pos:(pos + raw_samples)]
                raw_words = event_arr[pos:(pos + raw_samples)]

                # raw_data = np.zeros((2 * raw_words.size,), dtype=self.ADCWORDSIZE)
                # raw_data = np.zeros([evts, raw_samples], dtype=np.uint16)
                # raw_data[:, 0::2] = raw_words & 0xFFFF
                # raw_data[:, 1::2] = raw_words >> 16
                raw_data = np.zeros([raw_samples, evts], dtype=np.uint16)
                raw_data[0::2] = raw_words & 0xFFFF
                raw_data[1::2] = raw_words >> 16

                # data['raw_data'] = raw_data
                data['raw_data'] = raw_data.T  # Transpose this into rows
                pos += raw_samples

            maw_samples = current_event['maw_event_length']
            if maw_samples:
                # data['maw_data'] = event_arr[:, pos:(pos + maw_samples)]
                data['maw_data'] = event_arr[pos:(pos + maw_samples)].T

            return data, evts

        except Exception as e:
            # TODO: write to raw file and spit out error file
            print(e)

    def parse_to_struct(self, buffer, table_dtypes, *args): # TODO: Add save_first_raw_only
        """On the fly parser. Needs a buffer object of data, a dtype structure from a table (table._v_dtype),
         and the index of the channel. Returns a numpy record array necessary for fast saving to hdf5 file"""
        # 32 bit words
        if len(args) is 2:
            detector = hardware_constants.CHAN_TOTAL * args[0] + args[1]  # module and channel number
        else:
            detector = args  # detector number
        current_event = self.event_data[detector]
        event_length = current_event['event_length']//2  # 32 bit words, not 16

        if buffer.size is 0 or int(event_length) is 0:
            return

        # data_fields = ['format', 'det', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'pileup',
        # 'gate2', 'gate3', 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig',
        # 'maw_before_trig', 'en_start', 'en_max', 'raw_data', 'maw_data']

        # data_fields_dtypes = {'format':np.uint8, 'det': np.uint16, 'timestamp':np.uint64,
        # 'adc_max': np.uint16, 'adc_argmax': np.uint16, 'pileup': np.uint8, 'gate1': np.uint32,
        # 'gate2': np.uint32, 'gate3': np.uint32, 'gate4': np.uint32, 'gate5': np.uint32, 'gate6': np.uint32,
        # 'gate7': np.uint32, 'gate8': np.uint32, 'maw_max': np.uint32, 'maw_before_trig' : np.uint32,
        # 'maw_after_trig': np.uint32, 'en_start': np.uint32, 'en_max': np.uint32, 'raw_data': np.uint16,
        # 'maw_data': np.uint32, 'rid': np.uint32}

        try:
            evts = buffer.size//event_length

            data = np.zeros(evts, dtype=table_dtypes)  # This is a record (structured) numpy array

            event_arr = buffer.reshape([evts, event_length])
            # array where each row is an event

            data['rid'] = self.event_id + np.arange(evts)
            self.event_id += evts

            # Always get these
            ch_fmt = event_arr[:, 0] & 0xFFFF

            fmt = ch_fmt & 0b1111

            if np.max(fmt) is not np.min(fmt):
                pass  # TODO: ERROR. save to raw instead
            # data['format'] = fmt

            # ch = ch_fmt >> 4
            # data['channel'] = (ch & 0b1111)  # Hardwired
            # data['module'] = (ch >> 4)  # Writeable. Can be module number allowing for detector number
            data['det'] = ch_fmt >> 4

            data['timestamp'] = ((event_arr[:, 0].astype(np.uint64) & 0xFFFF0000) << 16) + event_arr[:, 1]

            pos = 2  # pointer to column where you are reading

            if bool(current_event['acc1_flag']):
                data['adc_max'] = event_arr[:, pos] & 0xFFFF
                data['adc_argmax'] = event_arr[:, pos] >> 16
                data['gate1'] = event_arr[:, (pos+1)] & 0xFFFFff  # 24 bits

                info = (event_arr[:, (pos+1)] >> (24 + 4))  # First 4 bits "reserved"
                data['pileup'] = (info & 0b1) | (info & 0b10)
                # data['repileup'] = info & 0b10  # Redundant information!
                # data['underflow'] = info & 0b100
                # data['overflow'] = info & 0b1000

                data['gate2'] = event_arr[:, pos + 2]
                data['gate3'] = event_arr[:, pos + 3]
                data['gate4'] = event_arr[:, pos + 4]
                data['gate5'] = event_arr[:, pos + 5]
                data['gate6'] = event_arr[:, pos + 6]
                pos += 7

            if bool(current_event['acc2_flag']):
                data['gate7'] = event_arr[:, pos]
                data['gate8'] = event_arr[:, pos+1]
                pos += 2

            if bool(current_event['maw_flag']):
                data['maw_max'] = event_arr[:, pos]
                data['maw_before_trig'] = event_arr[:, (pos + 1)]
                data['maw_after_trig'] = event_arr[:, (pos + 2)]
                pos += 3

            if bool(current_event['maw_energy_flag']):
                data['en_start'] = event_arr[:, pos]
                data['en_max'] = event_arr[:, (pos + 1)]
                pos += 2

            raw_samples = event_arr[:, pos] & 0x3FFffff  # in 32-bit words
            OxE_maw_status = event_arr[:, pos] >> 26
            OxE, maw_test, status_flag = OxE_maw_status >> 2, (OxE_maw_status >> 1 & 0b1), (OxE_maw_status & 0b1)
            # data['maw_select'] = maw_test  # Trigger MAW (0) or Energy MAW (1)
            # data['status'] = status_flag   # Pileup or Repileup
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
                raw_words = event_arr[:, pos:(pos + raw_samples)]
                # raw_data = np.zeros((2 * raw_words.size,), dtype=self.ADCWORDSIZE)
                raw_data = np.zeros([evts, raw_samples], dtype=np.uint16)
                raw_data[:, 0::2] = raw_words & 0xFFFF
                raw_data[:, 1::2] = raw_words >> 16
                data['raw_data'] = raw_data
                pos += raw_samples

            maw_samples = current_event['maw_event_length']
            if maw_samples:
                data['maw_data'] = event_arr[:, pos:(pos + maw_samples)]

            return data, evts

        except Exception as e:
            # TODO: write to raw file and spit out error file
            print(e)


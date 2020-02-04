import numpy as np
from common import hardware_constants
# from timeit import default_timer as timer


class parser(object):
    """This class performs on the fly parsing of data from SIS3316 card(s) based on set config settings"""

    def __init__(self, boards):
        self.boards = boards
        self.event_data = [c.event_stats for b in self.boards for c in b._chan]

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

    def parse16(self, buffer, *args):
        """On the fly parser. Needs a buffer object of data and the index of the channel. Returns a dictionary """
        # 16 bit words
        if len(args) is 2:
            detector = hardware_constants.CHAN_TOTAL * args[0] + args[1]  # module and channel number
        else:
            detector = args  # detector number
        current_event = self.event_data[detector]
        event_length = current_event['event_length']
        raw = np.frombuffer(buffer, dtype=self.ADCWORDSIZE)
        if raw.size is 0 or int(event_length) is 0:
            return

        # data_fields = ['format', 'channel', 'header', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'gate2', 'gate3',
        # 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig', 'maw_before_trig', 'en_max',
        #                'en_start', 'raw_data', 'maw_data']

        try:
            event_arr = raw.reshape([(raw.size//event_length), event_length])
            # array where each row is an event

            data = {}

            # Always get these
            ch_fmt = event_arr[:, 0]

            fmt = ch_fmt & 0b1111
            # TODO: Check that this matches what we expect?

            if np.max(fmt) is not np.min(fmt):
                pass  # TODO: ERROR. save to raw instead
            data['format'] = fmt

            ch = ch_fmt >> 4
            data['channel'] = (ch & 0b1111)  # Hardwired
            data['header'] = (ch >> 4)  # Writeable. Can be module number allowing for detector number

            data['timestamp'] = (event_arr[:, 1] << 32) + (event_arr[:, 2]) + (event_arr[:, 3] << 16)

            pos = 4  # pointer to column where you are reading

            if bool(current_event['acc1_flag']):
                data['adc_max'] = event_arr[:, 4]
                data['adc_argmax'] = event_arr[:, 5]
                data['gate1'] = event_arr[:, 6] + ((event_arr[:, 7] & 0xFF) << 16)

                info = (event_arr[:, 7] >> (8 + 4))  # First 4 bits "reserved"
                data['pileup'] = info & 0b1
                data['repileup'] = info & 0b10
                # data['underflow'] = info & 0b100
                # data['overflow'] = info & 0b1000

                data['gate2'] = event_arr[:, 8] + (event_arr[:, 9] << 16)
                data['gate3'] = event_arr[:, 10] + (event_arr[:, 11] << 16)
                data['gate4'] = event_arr[:, 12] + (event_arr[:, 13] << 16)
                data['gate5'] = event_arr[:, 14] + (event_arr[:, 15] << 16)
                data['gate6'] = event_arr[:, 16] + (event_arr[:, 17] << 16)
                pos += 14

            if bool(current_event['acc2_flag']):
                data['gate7'] = event_arr[:, pos] + (event_arr[:, (pos+1)] << 16)
                data['gate8'] = event_arr[:, (pos+2)] + (event_arr[:, (pos+3)] << 16)
                pos += 4

            if bool(current_event['maw_flag']):
                data['maw_max'] = event_arr[:, pos:] + (event_arr[:, (pos + 1)] << 16)
                data['maw_after_trig'] = event_arr[:, (pos + 2)] + (event_arr[:, (pos + 3)] << 16)
                data['maw_before_trig'] = event_arr[:, (pos + 4)] + (event_arr[:, (pos + 5)] << 16)
                pos += 6

            if bool(current_event['maw_energy_flag']):
                data['en_max'] = event_arr[:, pos] + (event_arr[:, (pos + 1)] << 16)
                data['en_start'] = event_arr[:, (pos + 2)] + (event_arr[:, (pos + 3)] << 16)
                pos += 4

            raw_samples = 2 * (event_arr[:, pos] + ((event_arr[:, pos+1] & 0x3FF) << 16))
            OxE_maw_status = event_arr[:, pos+1] >> 10
            OxE, maw_test, status_flag = OxE_maw_status >> 2, (OxE_maw_status >> 1 & 0b1), (OxE_maw_status & 0b1)
            # data['maw_select'] = maw_test  # Trigger MAW (0) or Energy MAW (1)
            # data['status'] = status_flag   # Pileup or Repileup
            pos += 2

            if np.min(OxE) is not np.max(OxE):  # Fast way to check all values are equal
                pass  # TODO: ERROR. save to raw instead

            # if raw_samp is not current_event['raw_event_length'] or np.min(raw_samp) is not np.max(raw_samp):
            #     pass

            if raw_samples:
                data['raw_data'] = event_arr[:, pos:(pos + raw_samples)]
                pos += raw_samples

            maw_samples = current_event['maw_event_length']
            if maw_samples:
                data['maw_data'] = event_arr[:, pos:(pos + maw_samples):2] \
                                   + (event_arr[:, (pos+1):(pos + maw_samples):2] << 16)

            return data

        except Exception as e:
            # TODO: write to raw file and spit out error file
            print(e)

    def parse32(self, buffer, *args):
        """On the fly parser. Needs a buffer object of data and the index of the channel. Returns a dictionary """
        # 32 bit words
        if len(args) is 2:
            detector = hardware_constants.CHAN_TOTAL * args[0] + args[1]  # module and channel number
        else:
            detector = args  # detector number
        current_event = self.event_data[detector]
        event_length = current_event['event_length']
        raw = np.frombuffer(buffer, dtype=self.FPGAWORDSIZE)
        if raw.size is 0 or int(event_length) is 0:
            return

        # data_fields = ['format', 'channel', 'header', 'timestamp', 'adc_max', 'adc_argmax', 'gate1', 'gate2', 'gate3',
        # 'gate4', 'gate5', 'gate6', 'gate7', 'gate8', 'maw_max', 'maw_after_trig', 'maw_before_trig', 'en_max',
        #                'en_start', 'raw_data', 'maw_data']

        try:
            event_arr = raw.reshape([(raw.size//(event_length//2)), (event_length//2)])
            # array where each row is an event

            data = {}

            # Always get these
            ch_fmt = event_arr[:, 0] & 0xFFFF

            fmt = ch_fmt & 0b1111
            # TODO: Check that this matches what we expect?

            if np.max(fmt) is not np.min(fmt):
                pass  # TODO: ERROR. save to raw instead
            data['format'] = fmt

            ch = ch_fmt >> 4
            data['channel'] = (ch & 0b1111)  # Hardwired
            data['header'] = (ch >> 4)  # Writeable. Can be module number allowing for detector number

            data['timestamp'] = ((event_arr[:, 0] & 0xFFFF0000) << 16) + event_arr[:, 1]

            pos = 2  # pointer to column where you are reading

            if bool(current_event['acc1_flag']):
                data['adc_max'] = event_arr[:, pos] & 0xFFFF
                data['adc_argmax'] = event_arr[:, pos] >> 16
                data['gate1'] = event_arr[:, (pos+1)] & 0xFFFFff  # 24 bits

                info = (event_arr[:, (pos+1)] >> (24 + 4))  # First 4 bits "reserved"
                data['pileup'] = info & 0b1
                data['repileup'] = info & 0b10
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
                data['maw_after_trig'] = event_arr[:, (pos + 1)]
                data['maw_before_trig'] = event_arr[:, (pos + 2)]
                pos += 3

            if bool(current_event['maw_energy_flag']):
                data['en_max'] = event_arr[:, pos]
                data['en_start'] = event_arr[:, (pos + 1)]
                pos += 2

            raw_samples = event_arr[:, pos] & 0x3FFffff  # in 32-bit words
            OxE_maw_status = event_arr[:, pos] >> 26
            OxE, maw_test, status_flag = OxE_maw_status >> 2, (OxE_maw_status >> 1 & 0b1), (OxE_maw_status & 0b1)
            # data['maw_select'] = maw_test  # Trigger MAW (0) or Energy MAW (1)
            # data['status'] = status_flag   # Pileup or Repileup
            pos += 1

            if np.min(OxE) is not np.max(OxE):  # Fast way to check all values are equal
                pass  # TODO: ERROR. save to raw instead

            # if (raw_samp * 2) is not current_event['raw_event_length'] or np.min(raw_samp) is not np.max(raw_samp):
            #     pass

            if raw_samples:  # TODO: Fix this mess
                raw_words = event_arr[:, pos:(pos + raw_samples)]
                # raw_data = np.zeros((2 * raw_words.size,), dtype=self.ADCWORDSIZE)
                raw_data = np.zeros((2 * raw_words.size,), dtype=np.uint16)
                raw_data[0::2] = raw_words & 0xFFFF
                raw_data[1::2] = raw_words >> 16
                data['raw_data'] = raw_data
                pos += raw_samples

            maw_samples = current_event['maw_event_length']
            if maw_samples:
                data['maw_data'] = event_arr[:, pos:(pos + maw_samples)]

            return data

        except Exception as e:
            # TODO: write to raw file and spit out error file
            print(e)

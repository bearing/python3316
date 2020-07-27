from common.registers import *
from channel import *
from group import *
from module_manager import *
from abc import abstractmethod, abstractproperty
from warnings import warn
from io import IOBase
import time


class destination(object):
    """ Proxy object. """
    target = None
    index = 0

    def __init__(self, target, skip=0):
        self.target = target
        self.index = skip

        if isinstance(target, self.__class__):
            self._return_target(target)

        elif isinstance(target, bytearray):
            self.push = self._push_bytearray

        elif isinstance(target, np.ndarray):
            self.push = self._push_numpy_array

        # elif isinstance(target, file): Python 2
        elif isinstance(target, IOBase):  # Python 3
            self.push = self._push_file

        # TODO: Add numpy array check

    @staticmethod
    def _return_target(target):
        return target

    def _push_numpy_array(self, source):  # source should be a byte array.
        # FIXME: This is clumsy
        limit = self.target.nbytes
        bytes_in_a_word = self.target.dtype.itemsize

        data = np.frombuffer(source, dtype=self.target.dtype)
        count = data.nbytes

        left_index = self.index  # in bytes
        right_index = left_index + count  # in bytes

        if right_index > limit:
            raise IndexError("Out of range.")

        self.target[left_index//bytes_in_a_word: right_index//bytes_in_a_word] = data
        self.index += count

    def _push_bytearray(self, source):
        limit = len(self.target)
        count = len(source)

        left_index = self.index
        right_index = left_index + count

        if right_index > limit:
            raise IndexError("Out of range.")

        self.target[left_index: right_index] = source
        self.index += count

    def _push_file(self, source):
        count = len(source)
        self.target.write(source)
        self.index += count
    # target.flush()?


class Sis3316(object):

    @abstractmethod
    def read(self, addr):
        pass

    @abstractmethod
    def write(self, addr, val):
        """ Execute general write request with a single parameter. """
        pass

    @abstractmethod
    def read_list(self, addrlist):
        """ Execute several read requests at once. """
        pass

    @abstractmethod
    def write_list(self, addrlist, datalist):
        """ Execute several write requests at once. """
        pass

    @property
    @abstractmethod
    def chan(self):
        pass

    @property
    @abstractmethod
    def hostname(self):
        pass

    def readout_buffer(self, chan_no, target_skip=0, chunksize=1024 * 1024, parse=False):  # This is the one used
        """Readout 1 channel buffer to a 1D bytearray object"""

        # ADCWORDSIZE = np.dtype(np.uint16).newbyteorder('<')  # Events are read as 16 bit words, small endian
        FPGAWORDSIZE = np.dtype(np.uint32).newbyteorder('<')  # Events are read as 32 bit words, small endian

        chan = self.chan[chan_no]
        bank = self.mem_prev_bank
        max_addr = chan.addr_prev

        finished = 0

        # ch_buffer = bytearray(max_addr * 4)
        ch_buffer = np.zeros(max_addr, dtype=FPGAWORDSIZE)

        dest = destination(ch_buffer, target_skip)

        while finished < max_addr:
            toread = min(chunksize, max_addr - finished)

            wtransferred = chan.bank_read(bank, dest, toread, finished)

            bank_after = self.mem_prev_bank
            max_addr_after = chan.addr_prev

            if bank_after != bank or max_addr_after != max_addr:
                raise self._BankSwapDuringReadExcept

            finished += wtransferred

        return ch_buffer

    def readout(self, chan_no, target, target_skip=0, chunksize=1024 * 1024):
        """ Returns ITERATOR. Useful for saving to raw binary file only. Yield returns status of readout"""

        # if opts is None:
        #    opts = {}
        # opts.setdefault('chunk_size', 1024 * 1024)  # words

        chan = self.chan[chan_no]
        bank = self.mem_prev_bank
        max_addr = chan.addr_prev

        finished = 0
        fsync = True  # the first byte in buffer is a first byte of an event

        dest = destination(target, target_skip)

        while finished < max_addr:
            toread = min(chunksize, max_addr - finished)

            wtransferred = chan.bank_read(bank, dest, toread, finished)

            bank_after = self.mem_prev_bank
            max_addr_after = chan.addr_prev

            if bank_after != bank or max_addr_after != max_addr:
                raise self._BankSwapDuringReadExcept

            finished += wtransferred

            yield {'transfered': wtransferred, 'sync': fsync, 'leftover': max_addr - finished}

            fsync = False

    def readout_pipe(self, chan_no, target, target_skip=0, swap_banks_auto=False, **kwargs):
        """ Readout generator. """
        # if opts is None:
        #    opts = {}
        # opts.setdefault('swap_banks_auto', False)

        while True:
            for retval in self.readout(chan_no, target, target_skip, **kwargs):
                yield retval

            if swap_banks_auto:
                self.mem_toggle()
            else:
                return

    def readout_last(self, chan_no, target, target_skip=0, **kwargs):
        """ Readout generator. Swap banks frequently. """
        self.mem_toggle()
        ret = self.readout(chan_no, target, target_skip, **kwargs)
        # return ret.next() # Python 2
        return next(ret)

    def poll_act(self, chanlist=None):
        """ Get a count of words in active bank for specified channels."""
        if chanlist is None:
            chanlist = []

        if not chanlist:
            chanlist = range(0, hardware_constants.CHAN_TOTAL - 1)

        data = []
        # TODO: make a single request instead of multiple .addr_actual property calls
        for i in chanlist:
            try:
                data.append(self.chan[i].addr_actual)
            except (IndexError, AttributeError):
                data.append(None)
        # End For
        return data

    def _event_stats(self):  # So not constantly having to wait for packets from each FPGA. Call once config set.
        # evt_lengths = np.zeros(hardware_constants.CHAN_TOTAL)
        format_evts = np.zeros([hardware_constants.CHAN_TOTAL, 7])  # 6 = len(chan.hit_events) + event_length
        try:
            for ind, chn in enumerate(self.chan):
                event_dict = chn.event_stats
                # evt_lengths[ind] = event_dict['event_length']
                format_evts[ind, :] = [event_dict['event_length'],
                                       event_dict['acc1_flag'],
                                       event_dict['acc2_flag'],
                                       event_dict['maw_flag'],
                                       event_dict['maw_max_values'],
                                       event_dict['raw_event_length'],
                                       event_dict['maw_event_length']
                                       ]

            # if np.count_nonzero(evt_lengths) is 0:
            if np.count_nonzero(format_evts[:, 0]) is 0:
                warn('No event lengths detected for any channel at IP: {f}. '
                     '\n Have you read a config file yet for that module?'.format(f=self.hostname), Warning)
                # TODO: Find subclass attribute hostname in a better way
            return format_evts
        except Exception as e:
            print(e)

    def _readout_status(self):
        """ Return current bank, memory threshold flags """
        data = self.read(SIS3316_ACQUISITION_CONTROL_STATUS)

        return {'armed': bool(get_bits(data, 16, 0b1)),
                'busy': bool(get_bits(data, 18, 0b1)),
                'threshold_overrun': bool(get_bits(data, 19, 0b1)),  # I.E. ANY FPGA
                'FP_threshold_overrun': bool(get_bits(data, 21, 0b1)),  # I.E. ANY Module if status lines enabled

                # more data than .addr_threshold - 512 kbytes. overrun is always True if .addr_threshold is 0!
                'bank': get_bits(data, 17, 0b1),

                'FPGA1_threshold_overrun': bool(get_bits(data, 25, 0b1)),  # Ch 1-4
                'FPGA2_threshold_overrun': bool(get_bits(data, 27, 0b1)),  # Ch 5-8
                'FPGA3_threshold_overrun': bool(get_bits(data, 29, 0b1)),  # Ch 9-12
                'FPGA4_threshold_overrun': bool(get_bits(data, 31, 0b1))   # Ch 13-16
                # ~ 'raw' : hex(data),
                }

    def disarm(self):
        """ Disarm sample logic."""
        self.write(SIS3316_KEY_DISARM, 0)

    def arm(self, bank=0):
        """ Arm sample logic. bank is 0 or 1. """
        if bank not in (0, 1):
            raise ValueError("'bank' should be 0 or 1, '{0}' given.".format(bank))

        if bank == 0:
            self.write(SIS3316_KEY_DISARM_AND_ARM_BANK1, 0)
        else:
            self.write(SIS3316_KEY_DISARM_AND_ARM_BANK2, 0)

    @property
    def mem_bank(self):
        """ Current memory bank. Return None if not armed."""
        stat = self._readout_status()
        if not stat['armed']:
            return None
        return stat['bank']

    @mem_bank.setter
    def mem_bank(self, value):
        self.arm(value)

    @property
    def mem_prev_bank(self):
        """ Previous memory bank. Return None if not armed."""
        bank = self.mem_bank
        if bank is None:
            return None

        return (bank - 1) % hardware_constants.MEM_BANK_COUNT

    def mem_toggle(self):
        """ Toggle memory bank (disarm and arm opposite) """
        current = self.mem_bank
        if current is None:
            raise self._NotArmedExcept

        new = current ^ 1
        self.arm(new)

    class _NotArmedExcept(Sis3316Except):
        """ Adc logic is not armed. """

    class _OverrunExcept(Sis3316Except):
        """  """

    class _BankSwapDuringReadExcept(Sis3316Except):
        """ Memory bank was swapped during readout. """

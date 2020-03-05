from common import hardware_constants
from common.utils import *
from common.registers import *
from triggers import adc_trigger
import numpy as np


class adc_channel(object):

    # evt_fields = ('Timestamp', '')  # TODO: Add fields for HDF5 parsing
    # Other fields that could be added: "Channel ID", "Module ID", "Format Bits"

    def __init__(self, container, l_id):
        self.board = container.board
        self.group = container
        self.gid = container.gid  # group index
        self.cid = l_id % hardware_constants.CHAN_PER_GRP  # channel index relative to FPGA
        self.idx = self.gid * hardware_constants.CHAN_PER_GRP + self.cid
        self.trig = adc_trigger(self, self.gid, self.cid)

    def bank_read(self, bank, dest, wcount, woffset=0):
        """ Read channel memory. """

        if woffset + wcount > hardware_constants.MEM_BANK_SIZE:
            raise ValueError("out of channel bound")

        if bank != 0 and bank != 1:
            raise ValueError("bank should be 0 or 1")

        if bank == 1:
            woffset += 1 << 24  # Bank select

        if self.cid % 2 == 1:
            woffset += 1 << 25  # Channel location in bank address space

        if self.cid < 2:
            mem_no = 0
        else:
            mem_no = 1

        return self.board.read_fifo(dest, self.gid, mem_no, wcount, woffset)

    def bank_poll(self, bank):
        """ Get number of bytes we can read. """
        # TODO
        return

    @property
    def dac_offset(self):
        """ Get ADC offsets (DAC) via SPI. """
        reg = SIS3316_ADC_GRP(DAC_OFFSET_READBACK_REG, self.gid)
        # offset = 0
        # mask = 0xFFFF
        # return self.board._get_field(reg, offset, mask)
        return self.board.read(reg)  # FIXME: Data format is off from documentation
        # raise AttributeError("You cant't read back loaded offset value.")

    @dac_offset.setter
    def dac_offset(self, value):
        """ Configure ADC offsets (DAC) via SPI. """
        reg = SIS3316_ADC_GRP(DAC_OFFSET_CTRL_REG, self.gid)
        chanmask = 0x3 & self.cid
        mask = 0xFFFF

        if value & ~mask:
            raise ValueError("Offset value is int 0...65535.")

        # We have a single DAC chip in chain, so using software LDACs.
        magic = [0x88f00011,
                 # 0x8-Write| 0x8-Set dcen/ref| 0xf-All|0x0|0x0|0x0|0x1 Standalone mode, Internal reference on | 0x1
                 # - Mysterious bit
                 0x85000000 + (chanmask << 20) + (0x1 << 4),  # Clear Code Register, 1 = Clears to Code 0x8000
                 0x82000000 + (chanmask << 20) + (value << 4),  # 0x8-Write| 0x2-Write to n, update all (soft LDAC)|...
                 ]
        for spell in magic:
            self.board.write(reg, spell)
            # ~ print hex(spell)
            usleep(10)  # Doc.: The logic needs approximately 7 usec to execute a command.

    @property
    def termination(self):
        """ Switch On/Off 50 Ohm terminator resistor on channel input. """
        reg = SIS3316_ADC_GRP(ANALOG_CTRL_REG, self.gid)
        offset = 2 + 8 * self.cid
        val = self.board._get_field(reg, offset, 0b1)
        return not bool(val)  # 1 means "disable termination"s

    @termination.setter
    def termination(self, enable):
        reg = SIS3316_ADC_GRP(ANALOG_CTRL_REG, self.gid)
        offset = 2 + 8 * self.cid
        val = not bool(enable)
        self.board._set_field(reg, val, offset, 0b1)

    @property
    def gain(self):
        """ Switch channel gain: 0->5V, 1->2V, 2->1.9V. """
        reg = SIS3316_ADC_GRP(ANALOG_CTRL_REG, self.gid)
        offset = 8 * self.cid
        return self.board._get_field(reg, offset, 0b11)

    @gain.setter
    def gain(self, value):
        if value & ~0b11:
            raise ValueError("Gain switch is a two-bit value.")
        reg = SIS3316_ADC_GRP(ANALOG_CTRL_REG, self.gid)
        offset = 8 * self.cid
        self.board._set_field(reg, value, offset, 0b11)

    ch_flags = ('invert',  # 0
                'intern_sum_trig',  # 1
                'intern_trig',  # 2
                'extern_trig',  # 3
                'intern_gate1',  # 4
                'intern_gate2',  # 5
                'extern_gate',  # 6
                'extern_veto',  # 7
                )

    @property
    def flags(self):
        """ Get/set channel flags (only all at once for certainty).
        The flags are listed in ch_flags attribute.
        """
        reg = SIS3316_ADC_GRP(EVENT_CONFIG_REG, self.gid)
        offset = 8 * self.cid
        data = self.board._get_field(reg, offset, 0xFF)

        ret = []
        for i in np.arange(8):
            if get_bits(data, i, 0b1):
                ret.append(self.ch_flags[i])
        return ret

    @flags.setter
    def flags(self, flag_list):
        reg = SIS3316_ADC_GRP(EVENT_CONFIG_REG, self.gid)
        offset = 8 * self.cid
        data = 0
        for flag in flag_list:
            shift = self.ch_flags.index(flag)
            data = set_bits(data, True, shift, 0b1)
        self.board._set_field(reg, data, offset, 0xFF)

    #  @property
    #  def event_maw_ena(self):
    #      """ Save MAW test buffer in event. """
    #      reg = SIS3316_ADC_GRP(DATAFORMAT_CONFIG_REG, self.gid)
    #      offset = 4 + 8 * self.cid
    #      return self.board._get_field(reg, offset, 0b1)

    #  @event_maw_ena.setter
    #  def event_maw_ena(self, enable):
    #      reg = SIS3316_ADC_GRP(DATAFORMAT_CONFIG_REG, self.gid)
    #      offset = 4 + 8 * self.cid
    #      self.board._set_field(reg, bool(enable), offset, 0b1)

    #  @property
    #  def event_maw_select(self):
    #      """ FIR MAW (0) or Energy MAW (1) """
    #      reg = SIS3316_ADC_GRP(DATAFORMAT_CONFIG_REG, self.gid)
    #      offset = 5 + 8 * self.cid
    #      return self.board._get_field(reg, offset, 0b1)

    #  @event_maw_select.setter
    #  def event_maw_select(self, enable):
    #      reg = SIS3316_ADC_GRP(DATAFORMAT_CONFIG_REG, self.gid)
    #      offset = 5 + 8 * self.cid
    #      self.board._set_field(reg, bool(enable), offset, 0b1)

    hit_flags = ('Save Peak High and Accum 1-6',  # 0
                 'Accum 7, 8',  # 1
                 'MAW Trigger Max, MAW Before Trigger, MAW After Trigger',  # 2
                 'Start and Max Energy MAW',  # 3
                 'MAW Test Buffer Enable',  # 4
                 'MAW Test Buffer Select',  # 5
                 )

    @property
    def format_flags(self):
        """Set the format and MAW flags in hit/event save data: 0-> peak high and accum1..6, 1-> accum7..8,
        2->MAW values, 3->Start/Max Energy MAW. Set all at once for certainty"""
        reg = SIS3316_ADC_GRP(DATAFORMAT_CONFIG_REG, self.gid)
        offset = 8 * self.cid
        mask = 0x3F
        data = self.board._get_field(reg, offset, mask)
        arr = np.zeros(len(self.hit_flags))

        for flg in np.arange(len(self.hit_flags)):
            arr[flg] = (data >> flg) & 0b1
        # return unpack_bits(data, len(self.hit_flags))
        # return np.array(self.hit_flags)[arr.astype(bool)]  # TODO: Fix Event Format to work with this
        return arr.astype(int)

    @format_flags.setter
    def format_flags(self, save_flag_list):
        reg = SIS3316_ADC_GRP(DATAFORMAT_CONFIG_REG, self.gid)
        offset = 8 * self.cid
        mask = 0x3F
        arr = np.array(save_flag_list)
        data = int(np.sum(arr * (2 ** np.arange(arr.size))))
        # data = pack_bits(save_flag_list)
        self.board._set_field(reg, data, offset, mask)

    # @property
    # def event_format_mask(self):
    #    """ Get event length: 0-> peak high and accum1..6, 1-> accum7..8, 2->MAW values, 3->reserved' """
    #    reg = SIS3316_ADC_GRP(DATAFORMAT_CONFIG_REG, self.gid)
    #    offset = 8 * self.cid
    #    mask = 0xF
    #    return self.board._get_field(reg, offset, mask)

    # @event_format_mask.setter
    # def event_format_mask(self, value):
    #    reg = SIS3316_ADC_GRP(DATAFORMAT_CONFIG_REG, self.gid)
    #    offset = 8 * self.cid
    #    mask = 0xF
    #    if value & ~mask:
    #        raise ValueError("A mask of the value is {0}. '{1}' given".format(hex(mask), value))
    #    self.board._set_field(reg, value, offset, mask)

    @property
    def event_length(self):
        """ Calculate the current size of the event (in 16 bit words). """
        emask = self.format_flags
        nraw = self.group.raw_window
        nmaw = self.group.maw_window
        maw_ena = emask[4]

        elen = 6 + nraw  # two header fields, 0xE field

        if maw_ena:
            elen += nmaw * 2

        # if emask & 0b1:
        if emask[0]:
            elen += 14  # peaking, accum 1..6

        # if emask & 0b10:
        if emask[1]:
            elen += 4  # accum 7,8

        if emask[2]:
            elen += 6  # maw values

        if emask[3]:
            elen += 4  # start energy value and max energy value

        return elen

    @property
    def event_stats(self):  # More full event save stats than event_length
        """ Calculate the current size of the event (in 16 bit words). """
        emask = self.format_flags
        nraw = self.group.raw_window
        nmaw = self.group.maw_window
        maw_ena = emask[4]

        elen = 6 + nraw  # two header fields, 0xE field

        if maw_ena:
            elen += nmaw * 2

        # if emask & 0b1:
        if emask[0]:
            elen += 14  # peaking, accum 1..6

        # if emask & 0b10:
        if emask[1]:
            elen += 4  # accum 7,8

        if emask[2]:
            elen += 6  # maw values

        if emask[3]:
            elen += 4  # start energy value and max energy value

        return {'event_length': elen,  # TODO: MULTIPLY BY ENABLED
                'raw_event_length': nraw,
                'maw_event_length': 2 * nmaw * maw_ena,
                'acc1_flag': emask[0],
                'acc2_flag': emask[1],
                'maw_flag': emask[2],
                'maw_energy_flag': emask[3]
                }

    # Not Used
    @property
    def intern_trig_delay(self):
        """ Delay of the internal trigger."""
        reg = SIS3316_ADC_GRP(INTERNAL_TRIGGER_DELAY_CONFIG_REG, self.gid)
        offset = 8 * self.cid
        mask = 0xFF
        return 2 * self.board._get_field(reg, offset, mask)

    @intern_trig_delay.setter
    def intern_trig_delay(self, value):
        reg = SIS3316_ADC_GRP(INTERNAL_TRIGGER_DELAY_CONFIG_REG, self.gid)
        offset = 8 * self.cid
        mask = 0x1FE  # the registry data = 2 * value
        if value & ~mask:
            raise ValueError("A mask of the value is {0}. '{1}' given".format(hex(mask), value))
        self.board._set_field(reg, value / 2, offset, mask / 2)
    # Not used

    _auto_properties = {
        'addr_actual': Param(0xffFFFF, 0, ACTUAL_SAMPLE_ADDRESS_REG, """ The actual sampling address 
        for the given channel. points to 32-bit words."""),
        'addr_prev': Param(0xffFFFF, 0, PREVIOUS_BANK_SAMPLE_ADDRESS_REG, """ The stored next sampling 
        address of the previous bank. It is the stop address + 1; points to 32-bit words."""),
        'en_peaking_time': Param(0xfFF, 0, FIR_ENERGY_SETUP_REG, """Peaking time: number of 
            samples to sum  with trapezoidal filter for energy measurement"""),
        'en_gap_time': Param(0x3FF, 12, FIR_ENERGY_SETUP_REG, """Gap time: number of 
            samples to skip with trapezoidal filter for energy measurement"""),
        'tau_factor': Param(0x3F, 24, FIR_ENERGY_SETUP_REG, """Tau (decimation) factor deconvolves 
            pre-amp decay"""),
        'tau_table': Param(0b11, 30, FIR_ENERGY_SETUP_REG, """Also used to set Tau, see other 
            documentation"""),  # TODO: Convert following cpp file to python. Add extra filter bit?
        # sis3316_energy_tau_factor_calculator.cpp
    }


# for name, prop in adc_channel._auto_properties.iteritems():  # Python 2
for name, prop in adc_channel._auto_properties.items():
    setattr(adc_channel, name, auto_property(prop, cid_offset=0x4))

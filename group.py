from channel import adc_channel
# from common.utils import SIS3316_ADC_GRP
from triggers import adc_trigger
from common.utils import *
from common.registers import *
from common import hardware_constants
import numpy as np


class adc_group(object):
    def __init__(self, container, id):
        self.gid = id % hardware_constants.CHAN_GRP_COUNT  # TODO: >>2
        self.idx = self.gid
        self.board = container
        self.channels = [adc_channel(self, i) for i in np.arange(4)]
        self.sum_trig = adc_trigger(self, self.gid, None)

    tap_delay_presets = {250: 0x48, 125: 0x48, 62.5: 0x0}

    def tap_delay_calibrate(self):  # TODO: REMOVE
        """ Calibrate the ADC FPGA input logic of the ADC data inputs.
        Doc.: A Calibration takes 20 ADC sample clock cycles.
        """
        self.board.write(SIS3316_ADC_GRP(INPUT_TAP_DELAY_REG, self.gid), 0xf00)

    def tap_delay_set(self):
        """ A coarse tuning of the tap delay (after calibration). """
        freq = self.board._freq
        data = self.tap_delay_presets[freq] | (0b11 << 8)  # select both ADC chips
        self.board.write(SIS3316_ADC_GRP(INPUT_TAP_DELAY_REG, self.gid), data)

    def clear_link_error_latch_bits(self):
        self.board.write(SIS3316_ADC_GRP(INPUT_TAP_DELAY_REG, self.gid), 0x400)

    @property
    def status(self):
        stat = self.board.read(SIS3316_ADC_GRP(STATUS_REG, self.gid))
        if stat != 0x130018:
            return stat
        return True

    @property
    def firmware_version(self):
        reg = SIS3316_ADC_GRP(FIRMWARE_REG, self.gid)
        data = self.board.read(reg)
        return {'type': get_bits(data, 16, 0xFFFF),
                'version': get_bits(data, 8, 0xFF),
                'revision': get_bits(data, 0, 0xFF)}

    @property
    def header(self):
        """ Channel's header (will be used in the event header)."""
        # Doc.: bits 11:4 are writeable
        #    	bits 3:2 has to be set with ADC FPGA group number -1
        #    	bits 1:0 (no write function, set to channel number in FPGA)
        return self.board._get_field(SIS3316_ADC_GRP(CHANNEL_HEADER_REG, self.gid), 24, 0xFF)

    @header.setter
    def header(self, value):
        if value is None:
            value = 0x0
    # def header(self, value=0x00):
        if value >> 8:
            raise ValueError("Single byte expected.")
        data = (value << 2) | self.gid
        self.board._set_field(SIS3316_ADC_GRP(CHANNEL_HEADER_REG, self.gid), data, 22, 0x3FF)

    # Currently unused (1 top)

    @property
    def scale(self):
        """ Set/get ADC input scale. Write to ADC chips via SPI. """
        # assume AD9643	#TODO: detect adc version (read chip ID). This is at chip address 0x01 with value 0x82
        reg_cmd = SIS3316_ADC_GRP(SPI_CTRL_REG, self.gid)
        reg_rdb = SIS3316_ADC_GRP(SPI_READBACK_REG, self.gid)

        ena = self.board._get_field(reg_cmd, 24, 0b1)

        cmd = 0xC0000000 + (0x18 << 8) + (ena << 24)
        magic = [cmd, cmd + 0x400000]  # ADC-1, ADC-2
        val = [None, None]

        for i in 0, 1:
            self.board.write(reg_cmd, magic[i])
            usleep(10)
            val[i] = 0xFF & self.board.read(reg_rdb)

        if val[0] != val[1]:
            # ~ print( "! scales are not the same: {0} and {1}".format(val[0], val[1]) )
            return None
        return val[0]

    @scale.setter
    def scale(self, value):

        # assume AD9643	#TODO: detect adc version (read chip ID)
        reg = SIS3316_ADC_GRP(SPI_CTRL_REG, self.gid)
        scales = {0xF: 2.087, 0xB: 1.992, 0x0: 1.75, 0x15: 1.50, 0x10: 1.383}

        if value not in scales:
            # translations = ['{0} => {1}V'.format(hex(k), v) for k, v in scales.iteritems()] # Python 2
            translations = ['{0} => {1}V'.format(hex(k), v) for k, v in scales.items()]
            raise ValueError("Scale preset value is one of {}.".format(translations))

        ena = self.board._get_field(reg, 24, 0b1)
        cmd = 0x80000000 + (0x18 << 8) + (ena << 24) + (value & 0x1F)  # 0x18 address controls input span range in
        # AD9643 manual. Default is 1.75 on power up. The minimum value is 1.383. Max is 2.087

        magic = [cmd, cmd + 0x400000,  # output enable, no invert, offset binary format
                 0x8100ff01, 0x8140ff01,  # write
                 ]

        for spell in magic:
            self.board.write(reg, spell)

    @property
    def test(self):
        """ Set/get ADC input scale. Write to ADC chips via SPI. """
        # assume AD9643	#TODO: detect adc version (read chip ID)
        reg_cmd = SIS3316_ADC_GRP(SPI_CTRL_REG, self.gid)
        reg_rdb = SIS3316_ADC_GRP(SPI_READBACK_REG, self.gid)

        ena = self.board._get_field(reg_cmd, 24, 0b1)

        cmd = 0xC0000000 + (0xD << 8) + (ena << 24)
        magic = [cmd, cmd + 0x400000]  # ADC-1, ADC-2
        val = [None, None]

        for i in 0, 1:
            self.board.write(reg_cmd, magic[i])
            usleep(10)
            val[i] = 0xF & self.board.read(reg_rdb)

        if val[0] != val[1]:
            # ~ print( "! scales are not the same: {0} and {1}".format(val[0], val[1]) )
            return None
        return val[0]

    @test.setter
    def test(self, value):

        # assume AD9643	#TODO: detect adc version (read chip ID)
        reg = SIS3316_ADC_GRP(SPI_CTRL_REG, self.gid)

        ena = self.board._get_field(reg, 24, 0b1)
        cmd = 0x80000000 + (0xD << 8) + (ena << 24) + (value & 0xF)

        magic = [cmd, cmd + 0x400000,  # output enable, no invert, offset binary format
                 0x8100ff01, 0x8140ff01,  # write
                 ]

        for spell in magic:
            self.board.write(reg, spell)

    # Currently unused (1 bot)

    @property
    def enable(self):
        """ Enable/disable adc output. """
        reg_cmd = SIS3316_ADC_GRP(SPI_CTRL_REG, self.gid)
        reg_rdb = SIS3316_ADC_GRP(SPI_READBACK_REG, self.gid)

        ena = self.board._get_field(reg_cmd, 24, 0b1)
        if not ena:
            return False

        # check adc configuration
        values = [(0xC1000000 + (0x08 << 8), 0x0),  # ADC-1, check not in standby
                  (0xC1000000 + (0x14 << 8), 0x4),  # ADC-1, output enable bar
                  (0xC1400000 + (0x08 << 8), 0x0),  # ADC-2
                  (0xC1400000 + (0x14 << 8), 0x4),  # ADC-2
                  ]

        for cmd, val in values:
            self.board.write(reg_cmd, cmd)
            usleep(10)
            data = 0xF & self.board.read(reg_rdb)
            # ~ print(hex(cmd), '->', hex(data))
            if data & 0xF != val:
                return False
        return True

    @enable.setter
    def enable(self, enble):
        reg = SIS3316_ADC_GRP(SPI_CTRL_REG, self.gid)
        # assume AD9643	#TODO: detect adc version (read chip ID)

        if enble:
            magic = [0x81001404, 0x81401404,  # output enable, no invert, offset binary format
                     0x8100ff01, 0x8140ff01,  # write
                     0x81000800, 0x81400800,  # normal operation, no standby
                     0x8100ff01, 0x8140ff01,  # write
                     ]

        else:
            magic = [0x81000802, 0x81400802,  # standby
                     0x8000ff01, 0x8040ff01,  # write
                     ]

        for spell in magic:
            self.board.write(reg, spell)
            usleep(10)  # Doc.: The logic needs approximately 7 usec to execute a command.

    @property
    def addr_threshold(self):
        """Doc.: The value will be compared with Actual Sample address counter (Bankx).
        Given in 32-bit words !
        """
        reg = SIS3316_ADC_GRP(ADDRESS_THRESHOLD_REG, self.gid)
        mask = 0xffFFFF
        return 4 * self.board._get_field(reg, 0, mask)

    @addr_threshold.setter
    def addr_threshold(self, value):
        reg = SIS3316_ADC_GRP(ADDRESS_THRESHOLD_REG, self.gid)
        mask = 0xffFFFF * 4
        if value & ~mask:
            raise ValueError("Words, not bytes! The mask is {0}. '{1}' given".format(hex(mask), value))
        self.board._set_field(reg, int(value / 4), 0, int(mask / 4))

    @property
    def gate_window(self):
        """ Doc.: The length of the Active Trigger Gate Window (2, 4, to 65536) """
        reg = SIS3316_ADC_GRP(TRIGGER_GATE_WINDOW_LENGTH_REG, self.gid)
        mask = 0xFFFF
        return 2 + self.board._get_field(reg, 0, mask)

    @gate_window.setter
    def gate_window(self, value):
        reg = SIS3316_ADC_GRP(TRIGGER_GATE_WINDOW_LENGTH_REG, self.gid)
        mask = 0xFFFF

        if value < 2:
            raise ValueError("Minimum gate length is 2. {0} given.".format(value))
        if value & ~mask:
            raise ValueError("The mask is {0}. '{1}' given".format(hex(mask), value))

        data = (value - 2) & mask  # 0xFFFE is 65536
        self.board._set_field(reg, data, 0, mask)

    @property
    def gate_intern_window(self):
        """Internal Gate Length."""
        reg = SIS3316_ADC_GRP(INTERNAL_GATE_LENGTH_CONFIG_REG, self.gid)
        mask = 0xFF
        offset = 8
        return 2 * self.board._get_field(reg, offset, mask)

    @gate_intern_window.setter
    def gate_intern_window(self, value):
        reg = SIS3316_ADC_GRP(INTERNAL_GATE_LENGTH_CONFIG_REG, self.gid)
        mask = 0x1FE
        offset = 8
        if value & ~mask:
            raise ValueError("The mask is {0}. '{1}' given".format(hex(mask), value))
        self.board._set_field(reg, value / 2, offset, mask / 2)

    @property
    def gate_coinc_window(self):
        """Internal Coincidence Gate Length."""
        reg = SIS3316_ADC_GRP(INTERNAL_GATE_LENGTH_CONFIG_REG, self.gid)
        mask = 0xFF
        offset = 0
        return 2 * self.board._get_field(reg, offset, mask)

    @gate_coinc_window.setter
    def gate_coinc_window(self, value):
        reg = SIS3316_ADC_GRP(INTERNAL_GATE_LENGTH_CONFIG_REG, self.gid)
        mask = 0x1FE
        offset = 0
        if value & ~mask:
            raise ValueError("The mask is {0}. '{1}' given".format(hex(mask), value))
        self.board._set_field(reg, value / 2, offset, mask / 2)

    _auto_properties = {
        'raw_start': Param(0xFFFe, 0, RAW_DATA_BUFFER_CONFIG_REG,
                           " The start index of the raw data buffer which will be copied to memory. "),
        'raw_window': Param(0xFFFe, 16, RAW_DATA_BUFFER_CONFIG_REG,
                            " The length of the raw data buffer which will be copied to memory. "),
        'pileup_window': Param(0xFFFe, 0, PILEUP_CONFIG_REG, " The window to recognize event pileup."),
        'repileup_window': Param(0xFFFe, 16, PILEUP_CONFIG_REG,
                                 """ The window to recognize trigger pileup."""),
        'delay': Param(0x3Fe, 0, PRE_TRIGGER_DELAY_REG,
                       "The number of samples before the trigger to save to memory. Max is 2042"),
        'delay_extra_ena': Param(True, 15, PRE_TRIGGER_DELAY_REG,
                                 "Turn on/off additional delay of FIR trigger (P+G)."),
        'maw_window': Param(0x3Fe, 0, MAW_TEST_BUFFER_CONFIG_REG, "MAW test buffer length. 0 to 1022."),
        'maw_delay': Param(0x3Fe, 16, MAW_TEST_BUFFER_CONFIG_REG,
                           "The number of MAW samples before the trigger to save to MAW test buffer. 2 to 1022."),

        'gate1_chan_mask': Param(0xF, 16, INTERNAL_GATE_LENGTH_CONFIG_REG,
                                 "Which channels included in gate-1."),
        'gate2_chan_mask': Param(0xF, 20, INTERNAL_GATE_LENGTH_CONFIG_REG,
                                 "Which channels included in gate-2."),

        'accum1_start': Param(0xFFFF, 0, ACCUMULATOR_GATE1_CONFIG_REG, "Accumulator-1 start index."),
        'accum2_start': Param(0xFFFF, 0, ACCUMULATOR_GATE2_CONFIG_REG, "Accumulator-2 start index."),
        'accum3_start': Param(0xFFFF, 0, ACCUMULATOR_GATE3_CONFIG_REG, "Accumulator-3 start index."),
        'accum4_start': Param(0xFFFF, 0, ACCUMULATOR_GATE4_CONFIG_REG, "Accumulator-4 start index."),
        'accum5_start': Param(0xFFFF, 0, ACCUMULATOR_GATE5_CONFIG_REG, "Accumulator-5 start index."),
        'accum6_start': Param(0xFFFF, 0, ACCUMULATOR_GATE6_CONFIG_REG, "Accumulator-6 start index."),
        'accum7_start': Param(0xFFFF, 0, ACCUMULATOR_GATE7_CONFIG_REG, "Accumulator-7 start index."),
        'accum8_start': Param(0xFFFF, 0, ACCUMULATOR_GATE8_CONFIG_REG, "Accumulator-8 start index."),

        'accum1_window': Param(0x1FF, 16, ACCUMULATOR_GATE1_CONFIG_REG, "Accumulator-1 length."),
        'accum2_window': Param(0x1FF, 16, ACCUMULATOR_GATE2_CONFIG_REG, "Accumulator-2 length."),
        'accum3_window': Param(0x1FF, 16, ACCUMULATOR_GATE3_CONFIG_REG, "Accumulator-3 length."),
        'accum4_window': Param(0x1FF, 16, ACCUMULATOR_GATE4_CONFIG_REG, "Accumulator-4 length."),
        'accum5_window': Param(0x1FF, 16, ACCUMULATOR_GATE5_CONFIG_REG, "Accumulator-5 length."),
        'accum6_window': Param(0x1FF, 16, ACCUMULATOR_GATE6_CONFIG_REG, "Accumulator-6 length."),
        'accum7_window': Param(0x1FF, 16, ACCUMULATOR_GATE7_CONFIG_REG, "Accumulator-7 length."),
        'accum8_window': Param(0x1FF, 16, ACCUMULATOR_GATE8_CONFIG_REG, "Accumulator-8 length."),
    }


# for name, prop in adc_group._auto_properties.iteritems(): # Python 2
for name, prop in adc_group._auto_properties.items():
    setattr(adc_group, name, auto_property(prop))

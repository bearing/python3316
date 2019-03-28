from common.registers import *
from common.utils import *  # Not required. Deals with issues on interpreters and PEP.
import os.path
import numpy as np
from common import hardware_constants
from abc import abstractmethod, ABCMeta
# from group import adc_group
import group  # TODO: Fix __all__ settings of adc_group
import channel
import triggers


class Sis3316(object):
    __metaclass__ = ABCMeta  # abstract class
    # __slots__ = ('groups', 'channels', 'triggers', 'sum_triggers')
    # __slots__ = ('grp', 'chan', 'trig', 'sum_triggers')  # TODO: Config and slave? Optimize with slots?

    def __init__(self):
        """ Initializes class structures, but not touches the device. """
        self.grp = [group.adc_group(self, i) for i in np.arange(hardware_constants.CHAN_GRP_COUNT)]
        # self.grp = [adc_group(self, i) for i in np.arange(hardware_constants.CHAN_GRP_COUNT)]
        self.chan = [c for g in self.grp for c in g.channels]
        self.trig = [c.trig for c in self.chan]
        self.sum_triggers = [g.sum_trig for g in self.grp]
        self.config = None
        # self.slave

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

    def _set_field(self, addr, value, offset, mask):
        """ Read value, set bits and write back. """
        data = self.read(addr)
        data = set_bits(data, value, offset, mask)
        self.write(addr, data)

    def _get_field(self, addr, offset, mask):
        """ Read a bitfield from register."""
        data = self.read(addr)
        return get_bits(data, offset, mask)

    _freq = None

    _freq_presets = {  # Si570 Serial Port 7PPM Registers (13, 14...)
        250: (0x20, 0xC2),
        125: (0x21, 0xC2),
        62.5: (0x23, 0xC2),
    }

    @property  # FP LVDS Bus Clock Driver. 0 -> Slave, 1-> Master
    def fp_driver(self):
        return self.fp_driver

    @fp_driver.setter
    def fp_driver(self, value):
        if value is None or 0 or 1:
            self.fp_driver = value
        else:
            ValueError('FP-Driver must be None, 0, or 1. {0} given.'.format(value))

    @property
    def freq(self):
        if self.fp_driver is None or 1:
            """ Program clock oscillator (Silicon Labs Si570) via I2C bus. """
            i2c = self.i2c_comm(self, SIS3316_ADC_CLK_OSC_I2C_REG)
            OSC_ADR = 0x55 << 1  # Slave Address, 0
            presets = self._freq_presets

            try:
                i2c.start()
                i2c.write(OSC_ADR)
                i2c.write(13)
                i2c.start()
                i2c.write(OSC_ADR | 0b1)

                reply = [0xFF & i2c.read() for i in range(0, 5)]
                reply.append(0xFF & i2c.read(ack=False))  # the last bit with no ACK

            except:
                i2c.stop()  # always send stop if something went wrong.

            i2c.stop()

            for freq, values in presets.iteritems():
                if values == tuple(reply[0:len(values)]):
                    self._freq = freq
                    return freq

            print 'Unknown clock configuration, Si570 RFREQ_7PPM values:', map(hex, reply)
        else:
            return None  # Return None since clock oscillator is not set via I2C bus (internal)

    @freq.setter  # So far only onboard oscillator and FP driver are defined. VXS panel and NIM input not implemented.
    def freq(self, value):  # value = (frequency, clock mode, master/slave)
        assert len(value) is 3, 'freq(values) requires 3 inputs: Clock freq., clock mode, ' \
                                                 'master/slave!'

        self.fp_driver = value[2]
        # self.clock_source = value[1]

        if self.fp_driver is not 0:  # Asynchronous operation (None) or FP Bus Master (1)
            if value[0] not in self._freq_presets:
                raise ValueError("Freq value is one of: {}".format(self._freq_presets.keys()))

            freq = value[0]
            i2c = self.i2c_comm(self, SIS3316_ADC_CLK_OSC_I2C_REG)
            OSC_ADR = 0x55 << 1  # Slave Address, 0
            presets = self._freq_presets

            try:
                set_freq_recipe = [
                    (OSC_ADR, 137, 0x10),  # Si570FreezeDCO
                    (OSC_ADR, 13, presets[freq][0], presets[freq][1]),  # Si570 High Speed/ N1 Dividers
                    (OSC_ADR, 137, 0x00),  # Si570UnfreezeDCO
                    (OSC_ADR, 135, 0x40),  # Si570NewFreq
                ]
                for line in set_freq_recipe:
                    i2c.write_seq(line)

            except:
                i2c.stop()  # always send stop if something went wrong.

            msleep(10)  # min. 10ms wait (according to Si570 manual)

        self._freq = value[0]

        # msleep(10)  # min. 10ms wait (according to Si570 manual)

        self.clock_source = value[1]

        _fp_driver_presets = {
            0: 0x2 + 0x4,  # enable FP status lines (0x2) and mystery bit from Struct (0x4)
            # TODO: Ask Stuck what this bit does?
            1: 0x2 + 0x4 + 0x0 + 0x1 + 0x10,  # onboard oscillator (0x0), enable FP control lines (0x1),
            # sample clock out as driver (0x10)
        }

        if self.fp_driver in _fp_driver_presets:
            self.write(SIS3316_FP_LVDS_BUS_CONTROL, _fp_driver_presets[self.fp_driver])


        self.write(SIS3316_KEY_ADC_CLOCK_DCM_RESET, 0)  # DCM Reset

        for grp in self.grp:
            grp.tap_delay_calibrate()
        usleep(10)

        for grp in self.grp:
            grp.tap_delay_set()
        usleep(10)

    # LEDs

    @property
    def leds(self):
        """ Get LEDs state. Returns 0 if LED is in application mode. """
        data = self.read(SIS3316_CONTROL_STATUS)
        status, appmode = get_bits(data, 0, 0b111), get_bits(data, 4, 0b111)
        return status & ~appmode  # 'on' if appmode[k] is 0 and status[k] is 1

    @leds.setter
    def leds(self, value):
        """ Turn LEDs on/off. """
        if value & ~0b111:
            raise ValueError("The state value is "
                             "a binary mask: 0...7 for 3 LEDs."
                             " '{0}' given.".format(value))
        self._set_field(SIS3316_CONTROL_STATUS, value, 0, 0b111)

    # LEDs

    @property
    def id(self):
        """ Module ID. """
        data = self.read(SIS3316_MODID)
        # noinspection PyTypeChecker
        return hex(data)

    @property
    def hardwareVersion(self):
        """ H/W version. """
        return self._get_field(SIS3316_HARDWARE_VERSION, 0, 0xF)

    @property
    def temp(self):
        """ Temperature  in degrees Celsius. """
        val = self._get_field(SIS3316_INTERNAL_TEMPERATURE_REG, 0, 0x3FF)
        if val & 0x200:  # 10-bit arithmetics
            val -= 0x400

        temp = val / 4.0
        return temp

    @property
    def serno(self):
        """ Serial No. """
        return self._get_field(SIS3316_SERIAL_NUMBER_REG, 0, 0xFFFF)

    @property
    def clock_source(self):
        """  Sample Clock Multiplexer. 0->onboard, 1->VXS backplane, 2->FP bus, 3-> NIM (with multiplier) """
        return self._get_field(SIS3316_SAMPLE_CLOCK_DISTRIBUTION_CONTROL, 0, 0b11)

    @clock_source.setter
    def clock_source(self, value):
        """  Set Sample Clock Multiplexer. """
        if value & ~0b11:
            raise ValueError("The value should integer in range 0...3. '{0}' given.".format(value))
        self._set_field(SIS3316_SAMPLE_CLOCK_DISTRIBUTION_CONTROL, value, 0, 0b11)

    @property
    def status(self):
        """ Status is True if everything is OK. """
        ok = True
        for grp in self.grp:
            grp.clear_link_error_latch_bits()
            status = grp.status
            if status != True:
                ok = False

        # check FPGA Link interface status
        self.write(SIS3316_VME_FPGA_LINK_ADC_PROT_STATUS, 0xE0E0E0E0)  # clear error Latch bits
        status = self.read(SIS3316_VME_FPGA_LINK_ADC_PROT_STATUS)
        if status != 0x18181818:
            ok = False

        return ok

    def reset(self):
        """ Reset the registers to power-on state."""
        self.write(SIS3316_KEY_RESET, 0)

    def ts_clear(self):
        """ Clear timestamp. """
        self.write(SIS3316_KEY_TIMESTAMP_CLEAR, 0)

# These are the configuration methods

    @staticmethod
    def _load_config_file(fname):
        if fname is None:
            ValueError('You must specify a filename to fname!')
        if isinstance(fname, basestring):
            if os.path.isfile(fname):
                with open(fname, 'r') as inf:
                    return eval(inf.read())
            else:
                raise ValueError('{fi} is not a path to a file'.format(fi=fname))
        else:
            raise ValueError('{fi} is not a string. It is a {ty}'.format(fi=fname, ty=type(fname).__name__))

    # _classes = {
    #     group.adc_group: 4,
    #     channel.adc_channel: 16,
    #     triggers.adc_trigger: 16
    # }

    # TODO: This is extremely clumsy. Might need to be moved to class and group.
    @staticmethod
    def parse_values(targets, prop_name, values, threshold=None):
        if type(values) is int:  # This is clumsy
            values = [values]

        try:
            val_array = np.array(values)
        except:
            raise ValueError('{v} is not a proper set of values in your config.'.format(v=values))

        assert val_array.size <= len(targets), "You are attempting to assign too many values to {}".format(prop_name)
        assert len(targets) % val_array.size is 0, "Number of values you are assigning to {} is not 1 or a power of 2"\
            .format(prop_name)
        repeat = len(targets) / val_array.size
        vals = np.repeat(val_array, repeat)  # np.repeat actually can accept float inputs, thus the assert
        # statement above

        # TODO: Check that all targets are of the same class? Is that necessary?

        try:
            for obj in targets:
                prop = obj.__dict__[prop_name]

                if not isinstance(prop, property):
                    raise TypeError('{!r} is not a property of class {}'.format(prop_name, type(obj).__name__))

                if vals[obj]:
                    prop.__set__(obj, vals[obj])
        except:
            raise ValueError('Failed to set {p} for {t} objects with: {v} \n'
                             'Did you mean to do this?'.format(p=prop_name, t=targets[0].__class__.__name__, v=values))

    def set_config(self, fname=None, FP_LVDS_Master=None):  # Default is assuming no FP communication
        self.config = self._load_config_file(fname)
        assert self.status, "Something is wrong with communication with the FPGAs or link layers on the card."
        # Everything is OK, clears error latch and link error latch bits
        self.reset()

        self.fp_driver = FP_LVDS_Master  # None = Asynchronous, 0 = FP Slave, 1 = FP Master Clock
        self.freq = (self.config['Clock Settings']['Clock Frequency'],
                     self.config['Clock Settings']['Clock Distribution'],
                     FP_LVDS_Master)

        for g in self.grp:
            g.enable = True  # I do this in a separate loop since this is re-enabling the ADCs, also there is a wait
            # assert g.enable, "Failed to communicate with ADC {fail}".format(fail=g)
            # TODO: Check this needs to be done or just write to bit 24 of SIS3316_ADC_CH1_4_SPI_CTRL_REG

        self.parse_values(self.chan, 'gain', self.config['Analog/DAC Settings']['Input Range Voltage'])
        self.parse_values(self.grp, 'header', self.config['Group Headers'])


















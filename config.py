from common.registers import *
from common.utils import *  # Not required. Deals with issues on interpreters and PEP.
import os.path
import numpy as np
from abc import abstractmethod


class Sis3316(object):
    class config(object):  # TODO: Potential issue with abstract methods not being to a parent class
        def __init__(self, fname=None, FP_LVDS_Bus_Slave=False):
            if FP_LVDS_Bus_Slave is True:
                self.slave = FP_LVDS_Bus_Slave
            else:
                self.slave = False

            if isinstance(fname, basestring):
                if os.path.isfile(fname):
                    with open(fname, 'r') as inf:
                        self.config = eval(inf.read())
                else:
                    raise ValueError('{fi} is not a path to a file'.format(fi=fname))
            else:
                raise ValueError('{fi} is not a string. It is a {ty}'.format(fi=fname, ty=type(fname).__name__))

            self.gid = np.arange(4)  # Group indexing
            self.cid = np.arange(4)  # Channel indexing

            self.clock_mode = self.config['Clock Settings']['Clock Distribution Control']
            if self.clock_mode is not 0 or 2:
                raise ValueError('Clock Distribution Control must be set to 0 or 2.')
            # TODO: Key Reset and Disarm Logic (set to power-up state and disable acquisition)

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

        def reset_and_disarm(self):
            self.write(SIS3316_KEY_RESET, 0)
            self.write(SIS3316_KEY_DISARM, 0)

        def clear_link_error_latch_bits(self):
            for grp in np.nditer(self.gid):
                self.write(SIS3316_ADC_GRP(SIS3316_ADC_CH1_4_INPUT_TAP_DELAY_REG, grp), 0x400)

        def clear_fpga_error_latch_bits(self):
            self.write(SIS3316_VME_FPGA_LINK_ADC_PROT_STATUS, 0xE0E0E0E0)

        def tap_delay_calibrate(self):
            for grp in np.nditer(self.gid):
                self.write(SIS3316_ADC_GRP(SIS3316_ADC_CH1_4_INPUT_TAP_DELAY_REG, grp), 0xf00)

        def tap_delay_set(self):
            """ A coarse tuning of the tap delay (after calibration). """
            freq = self._freq
            data = self.tap_delay_presets[freq] | (0b11 << 8)  # select both ADC chips
            for grp in np.nditer(self.gid):
                self.write(SIS3316_ADC_GRP(SIS3316_ADC_CH1_4_INPUT_TAP_DELAY_REG, grp), data)

        #  Clock settings
        _freq = None

        _freq_presets = {  # Si570 Serial Port 7PPM Registers (13, 14...)
            250: (0x20, 0xC2),
            125: (0x21, 0xC2),
            62.5: (0x23, 0xC2),
        }

        tap_delay_presets = {250: 0x48, 125: 0x48, 62.5: 0x0}

        @property  # TODO: FIX THIS
        def freq(self):
            if self.slave is False:  # Master clock using FP bus
                return self.freq_getter()
            else:
                pass

        @freq.setter  # TODO: FIX THIS
        def freq(self, value):
            if self.slave is False:  # Master clock using FP bus
                self.freq_setter(value)
            else:
                pass

        # @property
        # def freq(self):
        def freq_getter(self):
            """ Program clock oscillator (Silicon Labs Si570) via I2C bus. """
            i2c = self.i2c_comm(self, SIS3316_ADC_CLK_OSC_I2C_REG)  # Possibly unresolved
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

            # for freq, values in self._freq_presets.iteritems():
            for freq, values in presets.iteritems():
                if values == tuple(reply[0:len(values)]):
                    self._freq = freq
                    return freq

            print 'Unknown clock configuration, Si570 RFREQ_7PPM values:', map(hex, reply)

        # @freq.setter
        # def freq(self,value):
        def freq_setter(self, value):
            if value not in self._freq_presets:
                raise ValueError("Freq value is one of: {}".format(self._freq_presets.keys()))

            freq = value
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

            self._freq = value
            msleep(10)  # min. 10ms wait (according to Si570 manual)

            # FP LVDS Bus
            if self.clock_mode is 2:  # I will be explicit here for readability
                enable_FP_bus_status_lines = 0x2
                _tmp = 0x4  # TODO: Ask Struck what this does if anything (John's code had it but its not documented)
                FP_reg_data = enable_FP_bus_status_lines + _tmp

                if self.slave is False:  # TODO: Check if there is a master already first.
                    enable_FP_bus_control_lines = 0x1
                    sample_clock_driver_FP_bus = 0x10
                    # enable_LEMO_clock_in = 0x20  # If you wanted to do this, this is where it'd be done
                    FP_reg_data += enable_FP_bus_control_lines + sample_clock_driver_FP_bus
                self.write(SIS3316_FP_LVDS_BUS_CONTROL, FP_reg_data)
            # FP LVDS Bus

            self.write(SIS3316_KEY_ADC_CLOCK_DCM_RESET, 0)  # DCM Reset
            msleep(20)

            self.tap_delay_calibrate()
            # usleep(10)
            msleep(1)  # Struck waits this long. Not sure why.

            self.tap_delay_set()
            # usleep(10)
            msleep(1)  # Struck waits this long. Not sure why.

        # tap_delay_presets = {250: 0x48, 125: 0x48, 62.5: 0x0}

        def set_config(self):
            self.clear_link_error_latch_bits()
            self.clear_fpga_error_latch_bits()
            self.reset_and_disarm()h
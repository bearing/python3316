from channel import adc_channel
from common.utils import SIS3316_ADC_GRP
from common.registers import *
from common import hardware_constants
import numpy as np


class adc_group(object):
    def __init__(self, container, id):
        self.gid = id % hardware_constants.CHAN_GRP_COUNT  # TODO: >>2
        self.idx = self.gid
        self.board = container
        self.channels = [adc_channel(self, i) for i in np.arange(4)]
        # self.sum_trig = Adc_trigger(self, self.gid, None)

    tap_delay_presets = {250: 0x48, 125: 0x48, 62.5: 0x0}

    def tap_delay_calibrate(self):
        """ Calibrate the ADC FPGA input logic of the ADC data inputs.
        Doc.: A Calibration takes 20 ADC sample clock cycles.
        """
        self.write(SIS3316_ADC_GRP(SIS3316_ADC_CH1_4_INPUT_TAP_DELAY_REG, self.gid), 0xf00)

    def tap_delay_set(self):
        """ A coarse tuning of the tap delay (after calibration). """
        freq = self._freq
        data = self.tap_delay_presets[freq] | (0b11 << 8)  # select both ADC chips
        self.write(SIS3316_ADC_GRP(SIS3316_ADC_CH1_4_INPUT_TAP_DELAY_REG, self.gid), data)
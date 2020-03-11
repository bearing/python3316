from common import hardware_constants
from common.registers import *
from common.utils import *


# Param = namedtuple('param', 'mask, offset, reg, doc')

class adc_trigger(object):
    __slots__ = ('container', 'cid', 'gid', 'idx', 'board')  # Restrict attribute list (foolproof).

    def __init__(self, container, gid, cid):
        """ Trigger configuration. For sum triggers cid is None. """
        self.gid = gid
        if cid is None:
            self.cid = 4  # channel id for sum triggers is 4 (hardwired)
            self.idx = gid
        else:
            self.cid = cid
            self.idx = self.gid * hardware_constants.CHAN_PER_GRP + self.cid

        self.container = container
        self.board = container.board

    _auto_properties = {
        'maw_peaking_time': Param(0xfFF, 0, FIR_TRIGGER_SETUP_REG, """ Peaking time: number of 
            values to sum."""),
        'maw_gap_time': Param(0xfFF, 12, FIR_TRIGGER_SETUP_REG, """ Gap time (flat time)."""),
        'out_pulse_length': Param(0xFe, 24, FIR_TRIGGER_SETUP_REG, """ External NIM out pulse length 
            (stretched)."""),  # I.E. External Pulse in Clock Cycles after Cycle to drive other devices

        'threshold': Param(0xFffFFFF, 0, FIR_TRIGGER_THRESHOLD_REG,
                           """ Trapezoidal threshold value. \nThe full 27-bit running sum + 0x800 0000 is compared to 
                           this value to generate trigger."""),
        'cfd_ena': Param(0b11, 28, FIR_TRIGGER_THRESHOLD_REG,
                         """ Enable CFD with 50%. 0,1 - disable, 2 -zero crossing, 3 -enabled."""),
        'high_suppress_ena': Param(True, 30, FIR_TRIGGER_THRESHOLD_REG,
                                   """A trigger will be suppressed if the running sum of the trapezoidal filter goes 
                                   above the value of the High Energy Threshold register. \nThis mode works only with 
                                   CFD function enabled ! """),
        'enable': Param(True, 31, FIR_TRIGGER_THRESHOLD_REG, """ Enable trigger. """),

        'high_threshold': Param(0xFffFFFF, 0, FIR_HIGH_ENERGY_THRESHOLD_REG,
                                """ The full 27-bit running sum + 0x800 0000 is compared to the High Energy Suppress 
                                threshold value. \n Note 1: use channel invert for negative signals. """),
    }

    _conf_params = _auto_properties.keys()


# for name, prop in adc_trigger._auto_properties.iteritems(): # Python 2
for name, prop in adc_trigger._auto_properties.items():
    setattr(adc_trigger, name, auto_property(prop, cid_offset=0x10))


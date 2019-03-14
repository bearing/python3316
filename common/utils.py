from time import sleep
from hardware_constants import SIS3316_FPGA_ADC_GRP_REG_BASE, SIS3316_FPGA_ADC_GRP_REG_OFFSET
# from collections import namedtuple
import numpy as np


def SIS3316_ADC_GRP(reg, idx):
    ''' Select FPGA's register space according to group index. '''
    return reg + SIS3316_FPGA_ADC_GRP_REG_BASE + SIS3316_FPGA_ADC_GRP_REG_OFFSET * idx

# Param = namedtuple('param', 'mask, offset, reg, doc')
BITBUSY = 1 << 31


def msleep(x):
    sleep(x/1000.0)


def usleep(x):
    sleep(x/1000000.0)


def set_bits(int_type, val, offset, mask):
    """ Set bit-field with value."""
    data = int_type & ~(mask << offset)  # clear
    data |= (val & mask) << offset  # set
    return data


def get_bits(int_type, offset, mask):
    """ Get bit-field value according to mask and offset."""
    return (int_type >> offset) & mask


def _set_field(self, addr, value, offset, mask):
    """ Read value, set bits and write back. """
    data = self.read(addr)
    data = set_bits(data, value, offset, mask)
    self.write(addr, data)

def _get_field(self, addr, offset, mask):
    """ Read a bitfield from register."""
    data = self.read(addr)
    return get_bits(data, offset, mask)

def auto_property(param, cid_offset=0):
    """ Lazy coding. Generate class properties automatically.
    """
    if not isinstance(param, Param):
        raise ValueError("'param' is a namedtuple of type 'Param'.")

    def getter(self):
        reg = SIS3316_ADC_GRP(param.reg, self.gid)
        if cid_offset:
            reg += cid_offset * self.cid
        mask = param.mask
        offset = param.offset
        return self._get_field(reg, offset, mask)

    def setter(self, value):
        reg = SIS3316_ADC_GRP(param.reg, self.gid)
        if cid_offset:
            reg += cid_offset * self.cid
        mask = param.mask
        offset = param.offset
        if value & ~mask:
            raise ValueError("The mask is {0}. '{1}' given".format(hex(mask), value))
        self._set_field(reg, value, offset, mask)

    return property(getter, setter, None, param.doc)


class Sis3316Except(Exception):
    def __init__(self, *values, **kwvalues):
        self.values = values
        self.kwvalues = kwvalues

    def __str__(self):
        try:
            return self.__doc__.format(*self.values, **self.kwvalues)
        except IndexError:  # if arguments doesn't match format
            return self.__doc__

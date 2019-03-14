from common.utils import *
from common.utils import Sis3316Except  # Not required. Deals with issues on interpreters and PEP.

I2C_ACK = 1 << 8
I2C_START = 1 << 9
I2C_REP_START = 1 << 10
I2C_STOP = 1 << 11
I2C_WRITE = 1 << 12
I2C_READ = 1 << 13
I2C_BUSY = 1 << 31


class Sis3316(object):
    class i2c_comm(object):

        def __init__(self, container, register):
            self.container = container
            self.reg = register

        def write(self, byte_):
            """ Return True if got acknowledge. """
            if byte_ >> 8:
                raise ValueError('You can write per byte only.')

            self.container.write(self.reg, byte_ | I2C_WRITE)
            data = self.wait_busy()

            if data & I2C_ACK:
                return True
            else:
                return False

        def write_seq(self, bytes_):
            """ Write loop. """
            self.start()

            for byte in bytes_:
                if not self.write(byte):  # if have no ack
                    self.stop()
                    return False

            self.stop()
            return True

        def read(self, ack=True):
            if ack:
                cmd = I2C_READ | I2C_ACK
            else:
                cmd = I2C_READ

            self.container.write(self.reg, cmd)
            ret = self.wait_busy()
            return ret

        def start(self):
            self.container.write(self.reg, I2C_START)
            self.wait_busy()

        def stop(self):
            self.container.write(self.reg, I2C_STOP)
            self.wait_busy()

        def wait_busy(self):
            """ Return the register value. """
            count = 0
            while True:
                data = self.container.read(self.reg)
                if not data & I2C_BUSY:
                    return data

                count += 1
                if count > 10:
                    raise self._I2CHangExcept("I2c busy flag is set more than %d ms." % count)
                msleep(1)

        class _I2CHangExcept(Sis3316Except):
            """ I2c is busy too long. """

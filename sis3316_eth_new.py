# Newer versions of SIS3316 firmware have added a "packet identifier" byte

import abc
import socket
import select
# import sys
from struct import pack, unpack_from, error as struct_error
from random import randrange
from functools import wraps
from common.utils import *
from time import sleep
from common.utils import Sis3316Except  # Not required
from common.hardware_constants import *
from common.registers import *
import i2c
import module_manager
import readout
# import device
# import readout


def retry_on_timeout(f):
    """ Repeat action with a random timeout.
    You can configure it with an object's `.retry_max_count' and `.retry_max_timeout' properties.
    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        for i in range(0, self.retry_max_count):
            try:
                return f(self, *args, **kwargs)
            except self._TimeoutExcept:
                to = self.retry_max_timeout
                usleep(randrange(to / 2, to))

        raise self._TimeoutExcept(self.retry_max_count)

    return wrapper


class Sis3316(i2c.Sis3316, module_manager.Sis3316, readout.Sis3316):
    """ Ethernet implementation of sis3316 UDP-based protocol. The main functions are in interface and read_fifo
    """
    # Defaults:
    default_timeout = 0.1  # seconds
    retry_max_timeout = 100  # ms
    retry_max_count = 10
    jumbo = 4096  # set this to your ethernet's jumbo-frame size

    def __init__(self, host, port=5700):
        self.modname = host
        self.address = (host, port)
        self.cnt_wrong_addr = 0
        self._req_tot = 0  # Global counter of number of requests sent to card (for debugging purposes)
        self._nxt_id = 0  # Struct added a pkt ID byte to the UDP protocol (called request ID here).
        # FIFO reads now have a packet ID for each chunk and a request ID common to all chunks
        self._req_id = 0

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', port))
        # sock.setblocking(0)  #Python 2
        sock.setblocking(False)
        self._sock = sock

        for parent in self.__class__.__bases__:  # all parent classes
            parent.__init__(self)

    def __del__(self):
        """ Run this manually if you need to close socket."""
        self._sock.close()

    @classmethod
    def __status_err_check(cls, status):
        """ Interpret status field in response. """
        if status & 1 << 4:
            raise cls._SisNoGrantExcept
        if status & 1 << 5:
            raise cls._SisFifoTimeoutExcept
        if status & 1 << 6:
            raise cls._SisProtocolErrorExcept

    def check_recv_address(self, recvaddr):  # TODO: (NOTE 1) Check this works
        if self.cnt_wrong_addr < 100:  # Something is really wrong with the function or the ethernet if > 100
            if self.modname != recvaddr:
                self.cnt_wrong_addr += 1
            else:
                pass

    def cleanup_socket(self):
        """ Remove all data in the socket. """
        sock = self._sock
        bufsz = self.jumbo
        while 1:
            ready = select.select([sock], [], [], 0.0)
            if not ready[0]:
                break
            sock.recv(bufsz)

    def _req(self, msg):
        """ Send a request via UDP. """
        sock = self._sock

        # New Addition with New Firmware
        self._req_tot += 1
        if self._nxt_id >= 0xFF:  # pkt_id field in request is 1 byte
            self._req_id = 0xFF
            self._nxt_id = 0
        else:
            self._req_id = self._nxt_id
            self._nxt_id += 1
        # End of New Addition

        # Clean up if something is already there.
        garbage = select.select([sock], [], [], 0)
        if garbage[0]:
            self.cleanup_socket()
        sock.sendto(msg, self.address)

    def _resp_register(self, timeout=None):
        """ Get a single response packet. """
        if timeout is None:
            timeout = self.default_timeout

        sock = self._sock
        bufsz = self.jumbo
        response = None

        if select.select([sock], [], [], timeout)[0]:
            response, address = sock.recvfrom(bufsz)
        # TODO:check NOTE 1
        # if self.address != address
        # 	cnt_wrong_addr +=1
        #	pass

        if response:
            return response
        else:
            raise self._TimeoutExcept

    def _read_link(self, addr):
        """ Read request for a link interface. """
        msg = b''.join((b'\x10', pack('<BI', self._nxt_id, addr)))
        self._req(msg)
        resp = self._resp_register()

        try:  # Parse packet.
            hdr, rid, resp_addr, data = unpack_from('<BBII', resp)
            if hdr != 0x10 or resp_addr != addr:
                raise self._WrongResponseExcept

            if rid != self._req_id:
                raise self._PacketIDMismatchExcept(self._req_id, rid)

        except struct_error:
            raise self._MalformedResponseExcept
        return data

    def _write_link(self, addr, data):  # No Request Identifier Here
        """ Write request for a link interface. """
        msg = b''.join((b'\x11', pack('<II', addr, data)))
        self._req(msg)  # no ACK

    def _read_vme(self, addrlist):
        """ Read request on VME interface. """
        # print("Address List: ", addrlist)
        # print("Is int: ", all(isinstance(item, int) for item in addrlist))
        try:
            if not all(isinstance(item, (int, np.integer)) for item in addrlist):
                raise TypeError('_read_vme accepts a list of integers.')
        except:
            raise TypeError('_read_vme accepts a list of integers.')

        num = len(addrlist)
        if num == 0:
            return

        limit = VME_READ_LIMIT
        chunks = (addrlist,)
        if num > limit:
            # split addrlist by limit-sized chunks
            chunks = [addrlist[i:i + limit] for i in range(0, num, limit)]

        data = []
        for chunk in chunks:
            cnum = len(chunk)
            msg = b''.join((b'\x20', pack('<B', self._nxt_id), pack('<H%dI' % cnum, cnum - 1, *chunk)))
            self._req(msg)
            resp = self._resp_register()
            try:
                hdr, rid, stat = unpack_from('<BBB', resp[:3])
                if hdr != 0x20:
                    raise self._WrongResponseExcept
                if rid != self._req_id:
                    raise self._PacketIDMismatchExcept(self._req_id, rid)
                self.__status_err_check(stat)

                data.extend(unpack_from('<%dI' % cnum, resp[3:]))

            except struct_error:
                raise self._MalformedResponseExcept

        # end for
        return data

    def _write_vme(self, addrlist, datalist):
        """ Read request on VME interface. """

        try:
            if not all(isinstance(item, (int, np.integer)) for item in addrlist):
                raise TypeError('Address list must be a list of integers.')
            # if not all(isinstance(item, (int, long)) for item in datalist):  # Python2
            if not all(isinstance(item, (int, np.integer)) for item in datalist):
                raise TypeError('Data list must be list of integers')
        except:
            print("Datalist: ", datalist)
            print("Addrlist: ", addrlist)
            raise TypeError('Function accepts two lists of integers.')

        if len(addrlist) != len(datalist):
            raise ValueError('Two lists has to have equal size.')

        num = len(addrlist)
        if num == 0:
            return

        # Mix two lists: [addr1, data1, addr2, data2, ...]
        admix = [None, None] * num
        admix[::2] = addrlist
        admix[1::2] = datalist

        limit = VME_WRITE_LIMIT

        for idx in range(0, num, limit):
            ilen = min(limit, num - idx)

            msg = pack('<BBH%dI' % (2 * ilen,),
                       0x21, self._nxt_id, ilen - 1, *admix[2 * idx:2 * (idx + ilen)])
            self._req(msg)
            resp = self._resp_register()

            try:
                hdr, rid, stat = unpack_from('<BBB', resp)
                if hdr != 0x21:
                    raise self._WrongResponseExcept
                if rid != self._req_id:
                    raise self._PacketIDMismatchExcept(self._req_id, rid)
                self.__status_err_check(stat)

            except struct_error:
                raise self._MalformedResponseExcept

            except self._SisFifoTimeoutExcept:
                # we are not reading anything, so it's OK if FIFO-empty bit is '1'
                pass

    def open(self):
        """ Enable the link interface. """
        self._write_link(SIS3316_INTERFACE_ACCESS_ARBITRATION_CONTROL, 0x1)
        if not self._read_link(SIS3316_INTERFACE_ACCESS_ARBITRATION_CONTROL) & (1 << 20):  # if own grant bit not set
            raise IOError("Can't set Grant bit for Link interface")

    def close(self):
        """ Disable the link interface. """
        self._write_link(SIS3316_INTERFACE_ACCESS_ARBITRATION_CONTROL, 0x0)

    # ----------- Interface  ----------------------
    @retry_on_timeout
    def read(self, addr):
        """ Execute general read request with a single parameter. """
        if addr < 0x20:
            return self._read_link(addr)
        elif addr < 0x100000:
            return self._read_vme([addr])[0]
            # return self._read_vme(addr)[0]
        else:
            raise ValueError('Address {0} is wrong.'.format(hex(addr)))

    # @ In general it's not safe to retry write calls, so no retry_on_timeout here!
    def write(self, addr, word):
        if addr < 0x20:
            self._write_link(addr, word)
        elif addr < 0x100000:
            self._write_vme([addr], [word])
        else:
            raise ValueError('Address 0x%X is wrong.' % addr)

    def read_list(self, addrlist):
        """ Read a sequence of addresses at once. """
        # Check addresses.
        if any(addr / 0x100000 for addr in addrlist):  # any address is out of range
            raise ValueError('Some addresses are wrong.')

        if any(addr < 0x20 for addr in addrlist):
            raise NotImplementedError  # no sequential reads for link interface addresses.

        return retry_on_timeout(self.__class__._read_vme)(self, addrlist)

    def write_list(self, addrlist, datalist):
        """ Write to a sequence of addresses at once. """
        # Check addresses.
        # if not all(addr < 0x100000 for addr in addrlist):
        #    raise ValueError('Address {0} is wrong.'.format(hex(addr)))

        for addr in addrlist:
            if not all(addr < 0x100000):
                raise ValueError('Address {0} is wrong.'.format(hex(addr)))

        if not all(addr < 0x20 for addr in addrlist):
            raise NotImplementedError  # no sequential writes for link interface addresses.

        return self._write_vme(self, addrlist, datalist)
        # In general it's not safe to retry write calls, so no retry_on_timeout here!

    # ----------- FIFO stuff ----------------------
    def _ack_fifo_write(self, timeout=None):
        """ Get a FIFO write acknowledgement. """

        if timeout is None:
            timeout = self.default_timeout

        sock = self._sock
        packet_sz_bytes = 3  # Added pkt identifier
        bufzs = self.jumbo
        if select.select([sock], [], [], timeout)[0]:
            chunk, address = sock.recvfrom(bufzs)
            if len(chunk) == packet_sz_bytes:
                return chunk
            else:
                raise self._UnexpectedResponseLengthExcept(packet_sz_bytes, len(chunk))
        else:
            raise self._TimeoutExcept

    def _ack_fifo_read(self, dest, west_sz, timeout=None):  # TODO: This actually reads the data
        """
        Get response to FIFO read request.
        Args:
            dest: a buffer object which has a `push(smth)' method and an `index' property.
            west_sz: estimated count of words in response (to not to wait an extra timeout in the end).
        Returns:
            Nothing.
        Raise:
            _WrongResponceExcept, _UnorderedPacketExcept, _UnexpectedResponceLengthExcept
        """
        if timeout is None:
            timeout = self.default_timeout

        sock = self._sock
        header_sz_b = 3  # Changed from 2
        tempbuf = bytearray(self.jumbo)

        packet_idx = 0
        bcount = 0
        best_sz = west_sz * 4

        while select.select([sock], [], [], timeout)[0]:
            packet_sz, address = sock.recvfrom_into(tempbuf, self.jumbo)
            # packet_sz = sock.recv_into(tempbuf, self.jumbo)  # Do we care about address?

            # TODO:check address?
            # packet_sz, address = sock.recvfrom_into(tempbuf, self.jumbo)
            # if self.address != address
            # cnt_wrong_addr +=1
            # pass

            # Check that a packet is in order and it's status bits are ok.
            hdr = tempbuf[0]
            if hdr != 0x30:
                raise self._WrongResponseExcept("The packet header is not 0x30")

            rid = tempbuf[1]  # TODO: Check this works
            if rid != self._req_id:
                raise self._PacketIDMismatchExcept(self._req_id, rid)

            stat = tempbuf[2]
            self.__status_err_check(stat)

            packet_no = stat & 0xF
            if packet_no != packet_idx & 0xF:
                raise self._UnorderedPacketExcept

            packet_idx += 1
            # -- OK

            bcount += packet_sz - header_sz_b
            assert bcount <= best_sz, \
                "The length of response on FIFO-read request is %d bytes, but only %d bytes was expected." % (bcount,
                                                                                                              best_sz)
            assert bcount % 4 == 0, "data length in packet is not power of 4: %d" % (bcount,)
            dest.push(tempbuf[header_sz_b:packet_sz])  # TODO: Hopefully this works.
            if bcount == best_sz:
                return  # we have got all we need, so not waiting for an extra timeout
        raise self._TimeoutExcept

    def _fifo_transfer_read(self, grp_no, mem_no, woffset):  # TODO: This sends a request to read out memory
        """
        Set up fifo logic for read cmd.
        Args:
            grp_no: ADC index: {0,1,2,3}.
            mem_no: Memory chip index: {0,1}.
            woffset: Offset (in words).
        Returns:
            Address to read.
        Raises:
            _TransferLogicBusyExcept

        """
        if grp_no > 3:
            raise ValueError("grp_no should be 0...3")

        if mem_no != 0 and mem_no != 1:
            raise ValueError("mem_no is 0 or 1")

        reg_addr = SIS3316_DATA_TRANSFER_GRP_CTRL_REG + 0x4 * grp_no

        if self.read(reg_addr) & BITBUSY:
            raise self._TransferLogicBusyExcept(group=grp_no)

        # Fire "Start Read Transfer" command (FIFO programming)
        cmd = 0b10 << 30  # Read cmd
        cmd += woffset  # Start address

        if mem_no == 1:
            cmd += 1 << 28  # Space select bit

        self.write(reg_addr, cmd)  # Prepare Data transfer logic

    def _fifo_transfer_write(self, grp_no, mem_no, datalist, offset=0):  # Why would we do this?
        pass

    def _fifo_transfer_reset(self, grp_no):
        """ Reset memory transfer logic. """
        reg = SIS3316_DATA_TRANSFER_GRP_CTRL_REG + 0x4 * grp_no
        self.write(reg, 0)

    # ---------------------------

    def read_fifo(self, dest, grp_no, mem_no, nwords, woffset=0):  # TODO: I am here (12/24)
        """
        Get data from ADC unit's DDR memory.
        Readout is robust (retransmit on failure) and congestion-aware (adjusts an amount of data per request).
        Attrs:
            dest: an object which has a `push(smth)' method and an `index' property.
            grp_no: ADC group number.
            mem_no: memory unit number.
            nwords: number of words to read to dest.
            woffset: index of the first word.
        Returns:
            Number of words.
        """
        # TODO: make finished an argument by ref, so we can get the value even after Except

        fifo_addr = SIS3316_FPGA_ADC_GRP_MEM_BASE + grp_no * SIS3316_FPGA_ADC_GRP_MEM_OFFSET

        # Network congestion window:
        wcwnd_limit = FIFO_READ_LIMIT
        wcwnd = wcwnd_limit / 2
        wcwnd_max = wcwnd_limit / 2

        wmtu = 1440 / 4  # TODO: use mtu.py to determine MTU automatically (can be jumbo frames).
        #  This converts to 16-bit words

        wfinished = 0
        binitial_index = dest.index

        while wfinished < nwords:
            try:  # Configure FIFO
                self._fifo_transfer_reset(grp_no)  # cleanup
                self._fifo_transfer_read(grp_no, mem_no, int(woffset + wfinished))

            except self._WrongResponseExcept:  # some trash in socket
                self.cleanup_socket()
                sleep(self.default_timeout)
                continue

            except self._TimeoutExcept:
                sleep(self.default_timeout)
                continue  # FIXME: When to stop trying?

            # Data transmission
            while wfinished < nwords:

                try:
                    wnum = int(min(nwords - wfinished, FIFO_READ_LIMIT, wcwnd))

                    msg = b''.join((b'\x30', pack('<BHI', self._nxt_id, wnum - 1, fifo_addr)))
                    self._req(msg)
                    self._ack_fifo_read(dest, wnum)  # <- exceptions are most probable here

                    if wcwnd_max > wcwnd:  # recovery after congestion
                        wcwnd += (wcwnd_max - wcwnd) / 2

                    else:  # probe new maximum
                        wcwnd = int(min(wcwnd_limit, wcwnd + wmtu + (wcwnd - wcwnd_max)))

                except self._UnorderedPacketExcept:
                    # soft fail: some packets dropped
                    break

                except self._TimeoutExcept:
                    # hard fail (network congestion)
                    wcwnd_max = wcwnd
                    wcwnd = wcwnd / 2  # Reduce window by 50%
                    break

                finally:  # Note: executes before `break'
                    bfinished = (dest.index - binitial_index)
                    assert bfinished % 4 == 0, "Should read a four-byte words. %d, init %d" % (
                        bfinished, binitial_index)
                    wfinished = bfinished / 4

            # end while
            if wcwnd is 0:
                raise self._TimeoutExcept("many")

        # end while
        self._fifo_transfer_reset(grp_no)  # cleanup
        return wfinished

    def write_fifo(self, source, grp_no, mem_no, nwords, woffset=0):  # For the future, but why would we do this?
        pass

    # ----------- Exceptions ----------------------

    class _GarbageInSocketExcept(Sis3316Except):
        """ Socket is not empty. """

    class _MalformedResponseExcept(Sis3316Except):
        """ Response does not match the protocol. """

    class _WrongResponseExcept(Sis3316Except):
        """ Response does not match the request. {0}"""

    class _UnexpectedResponseLengthExcept(Sis3316Except):
        """ Was waiting for {0} bytes, but received {1}. """

    class _UnorderedPacketExcept(Sis3316Except):
        """ Ack packet not in right order. Probably some packets have been lost. """

    class _PacketsLossExcept(Sis3316Except):
        """ It looks like some packets have been lost. """

    class _WrongAddressExcept(Sis3316Except):
        """ Address {0} does not seem to make sense. """

    class _SisNoGrantExcept(Sis3316Except):
        """ sis3316 Link interface has no grant anymore. Use open() to request it."""

    class _SisFifoTimeoutExcept(Sis3316Except):
        """ sis3316 Access timeout during request (Fifo Empty). """

    class _SisProtocolErrorExcept(Sis3316Except):
        """ sis3316 Request command packet Except. """

    class _TimeoutExcept(Sis3316Except):
        """ Response timeout. Retried {0} times. """

    class _TransferLogicBusyExcept(Sis3316Except):
        """ Data transfer logic for unit #{group} is busy, or you forgot to do _fifo_transfer_reset. """

    class _PacketIDMismatchExcept(Sis3316Except):
        """ Expected Request ID: {0}. Received ID: {1} """


# You can run this file as a script for debug purposes.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('host', help='hostname or IP address')
    parser.add_argument('port', type=int, nargs="?", default=1234, help='UDP port number')
    args = parser.parse_args()
    # print(args.host, ' is string: ', isinstance(args.host, str))
    # print('Port: ', args.port)

    dev = Sis3316(args.host, args.port)
    print("mod ID:", hex(dev._read_link(0x4)))
    print("Hardware Version: ", hex(dev.hardwareVersion))
    dev.open()
    print("Temperature (Celsius): ", dev.temp)
    print("Serial Number: ", dev.serno)
    print("Start Up Frequency: ", dev.freq)
    # dev.configure()
    dev.freq = [250, 0, None]
    print("Set Frequency: ", dev.freq)


if __name__ == "__main__":
    import argparse
    main()

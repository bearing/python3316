#!/usr/bin/env python
# This is for everyone but Justin's cards

import socket, select
import sys
import argparse
import struct

# parse arguments
parser = argparse.ArgumentParser(description=
                                 'Test sis3316 network connection.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 # display default values in help
                                 )

parser.add_argument('host',
                    help='hostname or IP address')

parser.add_argument('port', type=int,
                    nargs="?", default=1234,  # optional
                    help='sis3316 destination port number')

args = parser.parse_args()
# ~ print args

# send message via UDP
server_address = (args.host, args.port)
print(server_address)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', args.port))
sock.setblocking(0)  # guarantee that recv will not block internally

msg = b'\x10\x01\x04\x00\x00\x00'  # request module_id, packet identifier = 1, register 4

try:
    sent = sock.sendto(msg, server_address)
    ready = select.select([sock], [], [],
                          0.5,  # timeout_in_seconds
                          )

    if ready[0]:
        resp, server = sock.recvfrom(1024)
        # print resp, server
        # print('raw response: ', resp.decode('hex_codec'))
        print('raw response: ', resp.hex())
        data = struct.unpack('<ccIHH', resp)

        # pkt_id = int.from_bytes(data[0],"little")
        print('OK', '( id:', hex(data[4]), ', rev:', hex(data[3]), ')')

    else:
        print("Fail: timed out.")
        print("Forgot to add mac address record to /etc/ethers and to run `arp -f'?")

except struct.error:
    print('Fail:', 'wrong format of response.')

except socket.gaierror as e:
    print('Fail:', str(e))

finally:
    sock.close()

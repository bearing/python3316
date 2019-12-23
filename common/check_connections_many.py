import socket, select
import sys
import argparse
import struct

# Run this program on terminal to just check that you can communicate with through ethernet to the cards AT ALL

# parse arguments
parser = argparse.ArgumentParser(description='Test sis3316 network connection.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 # display default values in help
                                 )

parser.add_argument('hosts', help='List of hostnames or IP addresses')

args = parser.parse_args()

ports = (6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6312)  # Hopefully this range
#  stays unused


# send message via UDP
def check_connection(mod_ip, port):
    server_address = (mod_ip, port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', port))
    # sock.setblocking(0) # Python 2
    sock.setblocking(False)  # guarantee that recv will not block internally

    msg = b'\x10\x04\x00\x00\x00'  # request module_id

    try:
        sent = sock.sendto(msg, server_address)
        ready = select.select([sock], [], [],
                              0.5,  # timeout_in_seconds
                              )

        if ready[0]:
            resp, server = sock.recvfrom(1024)
            # print resp, server
            # print('raw response: ', resp.encode('hex_codec')) Python 2
            print('raw response: ', resp.decode('hex_codec'))
            data = struct.unpack('<cIHH', resp)

            print('OK', '( id:', hex(data[3]), ', rev:', hex(data[2]), ')')

        else:
            print("Fail: timed out,")
            print("Hostname/IP: {f}".format(f=mod_ip))
            print("Port Number: {p}".format(p=port))
            # print "Forgot to add mac address record to /etc/ethers and to run `arp -f'?"

    except struct.error:
        print('Fail:', 'wrong format of response.')

    except socket.gaierror as e:
        print('Fail:', str(e))

    finally:
        sock.close()


mod_ips = args.hosts
for ind, mod in enumerate(mod_ips):
    print('mod_ips:', mod_ips)
    check_connection(mod, ports[ind])


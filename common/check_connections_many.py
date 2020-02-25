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
parser.add_argument('--ips', '-i', nargs='+', required=True, help='IP addresses of 3316 modules')
args = parser.parse_args()

ports = (6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6312)  # Hopefully this range
#  stays unused


# send message via UDP
def check_connection(mod_ip, port, legacy=False):
    server_address = (mod_ip, port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', port))
    # sock.setblocking(0) # Python 2
    sock.setblocking(False)  # guarantee that recv will not block internally

    if legacy:
        msg = b'\x10\x04\x00\x00\x00'  # older card messages
        data_types = '<cIHH'
    else:
        msg = b'\x10\x01\x04\x00\x00\x00'   # request module_id, packet identifier in newer software
        data_types = '<ccIHH'

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
            data = struct.unpack(data_types, resp)

            if legacy:
                print('OK', '( id:', hex(data[3]), ', rev:', hex(data[2]), ')')
            else:
                print('OK', '( id:', hex(data[4]), ', rev:', hex(data[3]), ')')
        else:
            print("Fail: timed out,")
            print("Hostname/IP: {f}".format(f=mod_ip))
            print("Port Number: {p}".format(p=port))
            # print "Forgot to add mac address record to /etc/ethers and to run `arp -f'?"

    except struct.error:
        if legacy:
            print('Fail:', 'wrong format of response.')
            sock.close()
        else:
            check_connection(mod_ip, port, legacy=False)

    except socket.gaierror as e:
        print('Fail:', str(e))

    finally:
        sock.close()


mods = args.ips
for ind, mod in enumerate(mods):
    print('mod_ip:', mod)
    check_connection(mod, ports[ind])
    print()


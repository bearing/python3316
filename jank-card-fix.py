import os
import socket
import struct
import select

gen_cmd = 'python data_subscriber.py -f sample_configs/CAMIS.json -i 192.168.1.{} -s raw_hdf5 -g 10000 -m 3 -sf Test >/dev/null 2>&1'

ips = [2, 3, 4, 5, 6, 7, 8, 9]
cards = [1, 2, 10, 12, 6, 7, 4, 5]

# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind(('', 1234))
# sock.setblocking(0)  # guarantee that recv will not block internally

#msg = b'\x10\x01\x04\x00\x00\x00'  # request module_id, packet identifier = 1, register 4

for i, ip in enumerate(ips):
    print("---------------------")
    print('Working on "fixing" the ip 192.168.1.{} (Card {})'.format(ip, cards[i]))
    # server_address = ('192.168.1.{}'.format(ip), 1234)
    #
    # sent = sock.sendto(msg, server_address)
    # ready = select.select([sock], [], [],
    #                       0.5,  # timeout_in_seconds
    #                       )
    #
    # if ready[0]:
    #     resp, server = sock.recvfrom(1024)
    #     # print resp, server
    #     # print('raw response: ', resp.decode('hex_codec'))
    #     data = struct.unpack('<ccIHH', resp)
    #
    #     # pkt_id = int.from_bytes(data[0],"little")
    #     print('OK', '( id:', hex(data[4]), ', rev:', hex(data[3]), ')')
    #     print('Connection to the card established!')
    #
    # else:
    #     print("Fail: timed out.")
    #     print("Forgot to add mac address record to /etc/ethers and to run `arp -f'?")

    os.system(gen_cmd.format(ip))

    print('"Fix" complete!')

print("---------------------")
print('This "fix" should have worked (hopefully)')
print('Try running the data_subscriber with all the cards now.')

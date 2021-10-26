import os

gen_cmd = 'python data_subscriber.py -f sample_configs/CAMIS.json -i 192.168.1.{} -s raw_hdf5 -g 1 -m 3 -sf Test >/dev/null 2>&1'

ips = [2, 3, 4, 5, 6, 7, 8, 9]
cards = [1, 2, 10, 12, 6, 7, 4, 5]

for i, ip in enumerate(ips):
    print("---------------------")
    print('Working on "fixing" the ip 192.168.1.{} (Card {})'.format(ip, cards[i]))
    os.system(gen_cmd.format(ip))
    print('"Fix" complete!')

print("---------------------")
print('This "fix" should have worked (hopefully)')
print('Try running the data_subscriber with all the cards now.')

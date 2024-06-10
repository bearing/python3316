import os
from datetime import datetime
import sys
now = datetime.now()

ips = {1: 3,
       2: 4,
       4: 5,
       6: 6,
       7: 7,
       8: 8,
       10: 9,
       12: 10,
       14: 3}

card_num = int(sys.argv[1])
fn = sys.argv[2]

df_name = '{}-{}-{}_{}'.format(now.month, now.day,
                                      str(now.year)[2:], fn)

run_cmd = ('python data_subscriber.py -f sample_configs/CAMIS.json -i '
            + '192.168.0.{ip} -s raw_hdf5 -m 30 --save_raw_waveforms '
            + '-sf CAMIS-DAQ-Testing/{df}').format(ip=ips[card_num], df=df_name)

os.system(run_cmd)

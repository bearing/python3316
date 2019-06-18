import sis3316_eth as sis
import i2c
import readout
import time
import data_subscriber as ds

t = time.time()
hostnames=['192.168.1.10']
dev = ds.daq_system(hostnames,configs='sample_configs/nsc.json')
dev.setup()
dev.subscribe(max_time=60, gen_time=1, save_fname='file1.dat')
'''
with open('file.dat','w') as file:
    for i in range(16):
        dev.modules[0].readout(i, file, target_skip=0)
        #time.sleep(0.5)
'''

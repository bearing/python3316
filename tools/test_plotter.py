import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

fname = '/Users/justinellin/repos/python_SIS3316/Data/example.bin'
# Use RadMapTest.json to test the damn MAW

dt = np.dtype(np.uint16)
num_bytes = dt.itemsize
dt = dt.newbyteorder('<')

waveform_num = 11
ev_len = 606
off = ev_len * num_bytes * waveform_num

print("Total number of events:", np.fromfile(fname, dtype=dt).size/ev_len)
arr = np.fromfile(fname, dtype=dt, count=ev_len, offset=off)  # Event Length in 16 bit words
# Offset is in Bytes!

format_flags = arr[0]
print("Format Flags: ")
print("Acc1: ", format_flags & 0b1, "| Acc2: ", format_flags & 0b10, "| MAWs: ", format_flags & 0b100,
      "| Energy MAWs: ", format_flags & 0b1000)
print("Channel ID: ", format_flags >>4)
print("Timestamp (seconds): ", ((arr[1]<<32) + (arr[2]) + (arr[3]<<16))/250000000)

# Justin
# print("Peak High:", arr[4], "| Index: ", arr[5])
# info = arr[7] >> 8  # Print
# print("Accumulator 1: ", arr[6] + ((arr[7] & 0xFF)<<16))
# print("Accumulator 2: ", arr[8] + (arr[9]<<16))
# print("Accumulator 3: ", arr[10] + (arr[11]<<16))
# print("Accumulator 4: ", arr[12] + (arr[13]<<16))
# print("Accumulator 5: ", arr[14] + (arr[15]<<16))
# print("Accumulator 6: ", arr[16] + (arr[17]<<16))
# print("Number of  Raw Samples: ", 2 * (arr[18] + ((arr[19] & 0x3FF) << 16)))
# OxE_maw_status = arr[19] >> 10
# OxE, maw_test, status_flag = OxE_maw_status >> 2, (OxE_maw_status >> 1 & 0b1), (OxE_maw_status & 0b1)
# print("0xE Check:", hex(OxE), "| Maw Test Flag: ", np.bool(maw_test), "| Status Flag: ", status_flag)
# samp_ind = 20  # start of raw samples

# EO Justin

# Chris

raw_samp = 2 * (arr[4] + ((arr[5] & 0x3FF) << 16))
print("Number of  Raw Samples: ", raw_samp)
OxE_maw_status = arr[5] >> 10
OxE, maw_test, status_flag = OxE_maw_status >> 2, (OxE_maw_status >> 1 & 0b1), (OxE_maw_status & 0b1)
print("0xE Check:", hex(OxE), "| Maw Test Flag: ", np.bool(maw_test), "| Status Flag: ", status_flag)
samp_ind = 6  # start of raw samples

# EO Chris

# Adam
# MAW_max = arr[4] + (arr[5]<<16)
# MAW_before= arr[6] + (arr[7]<<16)
# MAW_after= arr[8] + (arr[9]<<16)
# raw_samp = 2 * (arr[10] + ((arr[11] & 0x3FF) << 16))
# print("Number of  Raw Samples:", raw_samp)
# OxE_maw_status = arr[11] >> 10
# OxE, maw_test, status_flag = OxE_maw_status >> 2, (OxE_maw_status >> 1 & 0b1), (OxE_maw_status & 0b1)
# print("0xE Check:", hex(OxE), "| Maw Test Flag: ", np.bool(maw_test), "| Status Flag: ", status_flag)
# print("MAW Max:", MAW_max)
# print("MAW Before:", MAW_before)
# print("MAW After:", MAW_after)
# samp_ind= 12  # start of raw samples
# peaking_time = 5
# EO Adam

x = np.arange(raw_samp)  # 70 for PGI
plt.step(x * 4, arr[samp_ind:(samp_ind+raw_samp)])

plt.xlabel('Time (ns)')
plt.ylabel('ADC Value (arb. units)')
plt.title('Raw Waveform Number {n}'.format(n=waveform_num))
plt.show()

# If MAW enabled

dt_maw = np.dtype(np.uint32)
num_bytes_maw = dt_maw.itemsize
dt_maw = dt_maw.newbyteorder('<')

ev_len_maw = ev_len//2
off_maw = off//2

arr_maw = np.fromfile(fname, dtype=dt_maw, count=ev_len_maw, offset=off_maw)

maw_length = 400//2  # 32-bit words
x_maw = np.arange(maw_length)
maw_sample_ind = (samp_ind + raw_samp)//2
maw_WF = arr_maw[maw_sample_ind:(maw_sample_ind+maw_length)]
plt.step(x_maw * 4, maw_WF)

plt.xlabel('Time (ns)')
plt.ylabel('ADC Value (arb. units)')
plt.title('MAW Waveform Number {n}'.format(n=waveform_num))
plt.show()

# End of MAW enabled

# start_time = timer()
# bigarr = np.fromfile(fname, dtype=dt)
# print("Array Shape:", bigarr.shape)
# bigarr.reshape([bigarr.size,])
# time_elapsed = timer() - start_time
# print("Time Elapsed to Read:", time_elapsed)
# "Accumulators":
# "Gate 1": "Length": 9, Start Index": 0
# "Gate 2": "Length": 29, Start Index": 15, all others 0

import numpy as np
import matplotlib.pyplot as plt

fname = '/Users/justinellin/repos/python_SIS3316/Data/2020-01-20-1119Monday.bin'

dt = np.dtype(np.uint16)
dt = dt.newbyteorder('<')
waveform_num = 0
off = 360 * waveform_num
arr = np.fromfile(fname, dtype=dt, count=90, offset=off)  # Event Length in 16 bit words
# Offset is in Bytes! Offset = 360 for next event, etc.
# print("Array: ", arr)
format_flags = arr[0]
print("Format Flags: ")
print("Acc1: ", format_flags & 0b1, "| Acc2: ", format_flags & 0b10, "| MAWs: ", format_flags & 0b100,
      "| Energy MAWs: ", format_flags & 0b1000)
print("Channel ID: ", format_flags >>4)
print("Timestamp (seconds): ", ((arr[1]<<32) + (arr[2]<<16) + arr[3])/(250000000))
print("Peak High:", arr[4], "| Index: ", arr[5])
info = arr[7] >> 8  # Print
print("Accumulator 1: ", arr[6] + ((arr[7] & 0xFF)<<16))
print("Accumulator 2: ", arr[8] + (arr[9]<<16))
print("Accumulator 3: ", arr[10] + (arr[11]<<16))
print("Accumulator 4: ", arr[12] + (arr[13]<<16))
print("Accumulator 5: ", arr[14] + (arr[15]<<16))
print("Accumulator 6: ", arr[16] + (arr[17]<<16))
print("Number of  Raw Samples: ", 2 * (arr[18] + ((arr[19] & 0x3FF) << 16)))
OxE_maw_status = arr[19] >> 10
OxE, maw_test, status_flag = OxE_maw_status >> 2, (OxE_maw_status >> 1 & 0b1), (OxE_maw_status & 0b1)
print("0xE Check:", hex(OxE), "| Maw Test Flag: ", np.bool(maw_test), "| Status Flag: ", status_flag)
x = np.arange(70)
plt.step(x * 4, arr[20:])
plt.xlabel('Time (ns)')
plt.ylabel('ADC Value (arb. units)')
plt.title('Waveform Number {n}'.format(n=waveform_num))
plt.show()

# "Accumulators":
# "Gate 1": "Length": 9, Start Index": 0  # TODO: Print this out when file runs as well
# "Gate 2": "Length": 29, Start Index": 15, all others 0

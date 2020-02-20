
# Ethernet Data Aquisition Code for SIS3316 250MHz Digitizers 

(Work in Progress) SimpleSetup.docx goes over the basic setup in more detail

## Current Requirements

1. Python 3
2. Numpy

### `data_subscriber.py` output dictionary values (unordered)
The dictionary fields depend on the hit/event flags that are set in the config file. Unless otherwise stated, all data types are 32-bit uint. The FIR Trigger shaper is the "short shaper" and the FIR Energy shaper is "long shaper". 

### Fields
0) rid - global event id number, useful for sorting raw data events
1) timestamp - 64-bit uint in samples (depends on clock frequency)
2) channel - channel id, index starts at 0
3) header - writeable 12 bit field, if set to module id then can be used to get detector id with channel field
  
if 'Accumulator Gates 1-6 Flag' is set to True then the following fields are also present

4) adc_max - max adc value . 
5) adc_argmax - index of max value
6) pileup 
7) repileup
8) gate 1
9) gate 2
10) gate 3
11) gate 4
12) gate 5
13) gate 6

if 'Accumulator Gates 7-8 Flag' is set to True

14) gate 7
15) gate 8
  
if 'MAW Values Flag' is set to True:

16) maw_max - max value of fast shaper
17) maw_after_trig - fast shaper value 1 sample before trig
18) maw_before_trig - fast shaper value 1 sample after trig
  
if 'Energy Values Flag' is set to True:

19) en_start - long shaper value at trigger
20) en_max - max value of long shaper
  
if 'Save MAW Signal' is set to True:

21) maw_data - 'MAW Test Buffer Length' of 32-bit uint Samples from either the short or long shaper. Energy/Long MAW if 'Save MAW Signal' is true, Trigger/Short MAW otherwise
  
if 'Sample Length' > 0

22) raw_data - 'Sample Length' 16-bit uint ADC samples

## Notes

250 MHz -> 4 ns/sample

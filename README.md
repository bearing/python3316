
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
2) det - detector ID, first det is 0
  
if 'Accumulator Gates 1-6 Flag' is set to True then the following fields are also present

3) adc_max - max adc value . 
4) adc_argmax - index of max value
5) pileup 
6) repileup
7) gate 1
8) gate 2
9) gate 3
10) gate 4
11) gate 5
12) gate 6

if 'Accumulator Gates 7-8 Flag' is set to True

13) gate 7
14) gate 8
  
if 'MAW Values Flag' is set to True:

15) maw_max - max value of fast shaper
16) maw_after_trig - fast shaper value 1 sample before trig
17) maw_before_trig - fast shaper value 1 sample after trig
  
if 'Energy Values Flag' is set to True:

18) en_start - long shaper value at trigger
19) en_max - max value of long shaper
  
if 'Save MAW Signal' is set to True:

20) maw_data - 'MAW Test Buffer Length' of 32-bit uint Samples from either the short or long shaper. Energy/Long MAW if 'Save MAW Signal' is true, Trigger/Short MAW otherwise
  
if 'Sample Length' > 0

21) raw_data - 'Sample Length' 16-bit uint ADC samples

## Notes

250 MHz -> 4 ns/sample

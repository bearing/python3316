import json

blank_config = {
    # 'Number of Modules': None,
    'Module Info': {
        'Name': 'Justin_PGI',
        'Last 3 Serial Number Digits': 109,
        'ip address': '192.168.1.12'
    },
    'Clock Settings': {
        'Clock Frequency': 250,  # 250, 125, 62.5 (MHz)
        'Clock Distribution': 0,  # 0: Onboard Oscillator, 1: VXS-Bus Clock (not implemented) , 2: FP-LVDS-Bus
        # Clock, 3: External NIM Clock (not implemented)
        # 'FP-LVDS-Bus Master Module Name': None  # Now keyworded variable for set_config
    },
    'Analog/DAC Settings': {
        '50 Ohm Termination': True,  # Boolean. If disabled (0), termination is 1k
        'Input Range Voltage': 0,  # 0: 5V, 1: 1.9V, 2: 2V
        'DAC Offset': 0  # Max 16 bit  # Was 13000, now set to 32768
    },
    'Group Headers': 0,  # Max 8-bits
    'Hit Data': {  # This key is essential for on-the-fly parsing
        'Accumulator Gates 1-6 Flag': True,  # Boolean
        'Accumulator Gates 7-8 Flag': False,  # Boolean
        'MAW Values Flag': False,  # FIR Values: Max, before, and after trigger. With CFD enables high timing precision
        'Energy MAW Flag': False,  # Long shaper values. Start and Max.
        'MAW Test Buffer': False,
        'Save Raw Samples': False
    },
    'Trigger/Save Settings': {  # These are for  FIR (short) trigger filters, including sum FIR trigger settings
        'Trigger Gate Window': 100,  # Length in samples. You must define this  # Was 100, now 50
        'Sample Length': 70,  # Number of samples taken to generate triggering pulse  # Was 150, now 100
        'Sample Start Index': 0,  # Unless you know what this is, keep it at 0
        'Pre-Trigger Delay': 52,  # Samples saved before trigger, useful for baseline correction. Keep below 2042
        'Pre-Trigger P+G Bit': 0,  # adds peaking + gap time to previous value
        'Peaking Time': 10,  # Peaking Time in number of samples
        'Gap Time': 10,  # Number of samples
        'Pile Up': 75,
        'Re-Pile Up': 62,
        'CFD Enable': 3,  # 0,1: Disabled, 2: Zero Crossing, 3: 50% Crossing
        'High Energy Threshold': 0x08000000 + (10 * 50000),  # CFD Must be Enabled
        'Trigger Threshold Value': 0x08000000 + (10 * 900),  # Changed from 0xB0 to 40960
        'Sum Trigger CFD Enable': 3,
        'Sum Trigger High Energy Threshold': 0x08000000 + (10 * 50000),
        'Sum Trigger Peaking Time': 10,
        'Sum Trigger Gap Time': 10,
        'Sum Trigger Threshold Value': 0x08000000 + (10 * 480),
    },
    'MAW Settings': {
        'MAW Test Buffer Length': 100,  # Maw Values Flag must be set to 1
        'MAW Test Buffer Delay': 0,  # Same as above
        'MAW Test Buffer Select': 0  # (0, default): Save Short Shaper (FIR) MAW. (1) Save Energy MAW
    },
    'Energy Filter': {  # This is the longer filter used for pulse mode energy measurements
        'Peaking Time': None,
        'Gap Time': None,
        'Tau Factor': None,  # 1 of 2 values needed to deconvolve pre-amp decay
        'Tau Table': None  # 1 of 2 values needed to deconvolve pre-amp decay
    },
    'Event Settings': {  # These are all Booleans. Currently must be set for all (16) channels
        'Invert Signal': 1,  # 0 for positive polarity signals, 1 for negative
        'Sum Trigger Enable': 1,  # 0: Disable, 1: Enable Sum Triggers
        'Internal Trigger': 0,
        'External Trigger': 0,  # This would almost certainly need to be done for time correlated measurements
        'Internal Gate 1': 0,  # Not used yet
        'Internal Gate 2': 0,  # Not used yet
        'External Gate': 0,  # Not used yet
        'External Veto': 0,  # Not used yet
    },
    'Address Threshold': 200,  # The water level of the 4 FPGA memories before "memory threshold flags" are
    # triggered

    #  'Readout Settings': {  # Very important settings here. They will have to be set
    #      'Events': {
    #          'Water Level': None,  # Number of 32 bit words saved before flagging the bank is full
    #          'Keep Saving': None,  # Keep saving events up until the memory bank is swapped
    #      },  # Events mode performs a readout (near) when the water level is full
    #      'Time': None  # Time in seconds before a readout occurs
    #  },
    'Accumulators': {  # Hit/Event flags must be set for it to actually save them
        'Gate 1': {
            'Length': 0x9,
            'Start Index': 0x0
        },
        'Gate 2': {
            'Length': 0x1D,
            'Start Index': 0xF
        },
        'Gate 3': {
            'Length': None,
            'Start Index': None
        },
        'Gate 4': {
            'Length': None,
            'Start Index': None
        },
        'Gate 5': {
            'Length': None,
            'Start Index': None
        },
        'Gate 6': {
            'Length': None,
            'Start Index': None
        },
        'Gate 7': {
            'Length': None,
            'Start Index': None
        },

        'Gate 8': {
            'Length': None,
            'Start Index': None
        }
    }
}

with open('PGItest.json', 'w') as fp:
    json.dump(blank_config, fp, sort_keys=True, indent=4)

with open('PGItest.json', 'r') as fp:
    loaded_config = json.load(fp)

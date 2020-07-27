blank_config = {
    # 'Number of Modules': None,
    'Module Info': {
        'Name': None,
        'ip address': None
    },
    'Clock Settings': {
        'Clock Frequency': None,  # 250, 125, 62.5 (MHz)
        'Clock Distribution': None,  # 0: Onboard Oscillator, 1: VXS-Bus Clock (not implemented) , 2: FP-LVDS-Bus
        # Clock, 3: External NIM Clock (not implemented)
        # 'FP-LVDS-Bus Master Module Name': None  # Now keyworded variable for set_config
    },
    'Analog/DAC Settings': {
        '50 Ohm Termination': None,  # Boolean. If disabled (0), termination is 1k
        'Input Range Voltage': None,  # 0: 5V, 1: 1.9V, 2: 2V
        'DAC Offset': None  # Max 16 bit
    },
    # 'Group Headers': None,  # Max 8-bits
    'Hit/Event Data': {  # This key is essential for on-the-fly parsing
        'Accumulator Gates 1-6 Flag': None,  # Boolean
        'Accumulator Gates 7-8 Flag': None,  # Boolean
        'MAW Values Flag': None,  # Boolean
        'Energy Values Flag': None,  # Boolean
        'Save MAW Signal': None  # Boolean
        # 'Save Raw Samples': None  # This doesn't do anything. Determined by raw sample length
    },
    'Trigger/Save Settings': {  # These are for  FIR (short) trigger filters, including sum FIR trigger settings
        'Trigger Gate Window': None,  # Length in samples. You must define this
        'Sample Length': None,  # Number of samples taken to generate triggering pulse
        'Sample Start Index': 0,  # Unless you know what this is, keep it at 0
        'Pre-Trigger Delay': None,  # Samples saved before trigger, useful for baseline correction. Keep below 2042
        'Peaking Time': None,  # Peaking Time in number of samples
        'Gap Time': None,  # Number of samples
        'Pile Up': None,
        'Re-Pile Up': None,
        'CFD Enable': 0,  # 0,1: Disabled, 2: Zero Crossing, 3: 50% Crossing
        'High Energy Threshold': None,  # CFD Must be Enabled
        'Trigger Threshold Value': None,
        'Sum Trigger CFD Enable': None,
        'Sum Trigger High Energy Threshold': None,
        'Sum Trigger Peaking Time': None,
        'Sum Trigger Gap Time': None,
        'Sum Trigger Threshold Value': None,
    },
    'MAW Settings': {
        'MAW Test Buffer Length': None,  # Maw Values Flag must be set to 1
        'MAW Test Buffer Delay': None,  # Same as above
    },
    'Energy (Long Shaper) Filter': {  # This is the longer filter used for pulse mode energy measurements
        'Peaking Time': None,
        'Gap Time': None,
        'Tau Factor': None,  # 1 of 2 values needed to deconvolve pre-amp decay
        'Tau Table': None  # 1 of 2 values needed to deconvolve pre-amp decay

    },
    'Event Settings': {  # These are all Booleans. Currently must be set for all (16) channels
        'Invert Signal': None,  # 0 for positive polarity signals, 1 for negative
        'Sum Trigger Enable': None,  # 0: Disable, 1: Enable Sum Triggers
        'Internal Trigger': None,
        'External Trigger': 0,  # This would almost certainly need to be done for time correlated measurements
        'Internal Gate 1': 0,  # Not used yet
        'Internal Gate 2': 0,  # Not used yet
        'External Gate': 0,  # Not used yet
        'External Veto': 0,  # Not used yet
    },
    # ch_flags = ('invert',  # 0
    #            'intern_sum_trig',  # 1
    #            'intern_trig',  # 2
    #            'extern_trig',  # 3
    #            'intern_gate1',  # 4
    #            'intern_gate2',  # 5
    #            'extern_gate',  # 6
    #            'extern_veto',  # 7
    #            )


    #  'Readout Settings': {  # Very important settings here. They will have to be set
    #     'Readout Mode': None,  # 0: Events, 1: Time
    #      'Events': {
    #          'Water Level': None,  # Number of 32 bit words saved before flagging the bank is full
    #          'Keep Saving': None,  # Keep saving events up until the memory bank is swapped
    #      },  # Events mode performs a readout (near) when the water level is full
    #      'Time': None  # Time in seconds before a readout occurs
    #  },
    'Accumulator Gates': {  # Hit/Event flags must be set
        'Gate 1': {
            'Length': None,
            'Starting Index': None
        },
        'Gate 2': {
            'Length': None,
            'Starting Index': None
        },
        'Gate 3': {
            'Length': None,
            'Starting Index': None
        },
        'Gate 4': {
            'Length': None,
            'Starting Index': None
        },
        'Gate 5': {
            'Length': None,
            'Starting Index': None
        },
        'Gate 6': {
            'Length': None,
            'Starting Index': None
        },
        'Gate 7': {
            'Length': None,
            'Starting Index': None
        },

        'Gate 8': {
            'Length': None,
            'Starting Index': None
        }
    }
}

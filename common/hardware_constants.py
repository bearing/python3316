# ADC FPGA FIFO
SIS3316_FPGA_ADC_GRP_REG_BASE = 0x1000
SIS3316_FPGA_ADC_GRP_REG_OFFSET = 0x1000
SIS3316_FPGA_ADC_GRP_MEM_BASE = 0x100000
SIS3316_FPGA_ADC_GRP_MEM_OFFSET = 0x100000
VME_READ_LIMIT = 64  # (32-bit) words
VME_WRITE_LIMIT = 64  # (32-bit) words
# FIFO_READ_LIMIT = int(0x40000/4)  # bytes-> (32-bit) words
FIFO_READ_LIMIT = 0x40000//4  # bytes-> (32-bit) words
FIFO_WRITE_LIMIT = 256	 # (32-bit) words

# SIS3316 Specific Hardware
MEM_BANK_SIZE = 0x4000000  # 64MB
MEM_BANK_COUNT = 2
CHAN_GRP_COUNT = 4
CHAN_PER_GRP   = 4
CHAN_MASK  = CHAN_PER_GRP - 1  # 0b11
CHAN_TOTAL = CHAN_PER_GRP * CHAN_GRP_COUNT  # 16

# Truly Random Struck Decision
TRIG_THRESHOLD_OFFSET = 0x8000000  # For triggering purposes the trigger value + this number is compared to trap. filter

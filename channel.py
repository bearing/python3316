from common import hardware_constants


class adc_channel(object):
    def __init__(self, container, id):
        self.board = container.board
        self.group = container
        self.gid = container.gid  # group index
        self.cid = id % hardware_constants.CHAN_PER_GRP  # channel index
        self.idx = self.gid * hardware_constants.CHAN_PER_GRP + self.cid

        # self.trig = Adc_trigger(self, self.gid, self.cid)

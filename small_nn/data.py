class data():
    """docstring for data."""

    def __init__(self, arg):
        super(data, self).__init__()
        self.arg = arg


class batch():
    """docstring for batch."""

    def __init__(self, arg):
        super(batch, self).__init__()
        self.arg = arg
    x = []
    y = []

    def split(data, type):
        raise NotImplementedError

def load(data, numbbaches): # split data into batches
    type = data.type;
    dataLength = len(data)
    segmentSize = datalength/numbacches
    batchs = {}
    for i in range(0, dataLength, segmentSize):
        batchs["batch{0}".format(i)] = batch(data[i:i+segmentSize], type)

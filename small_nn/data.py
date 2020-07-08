import os
import numpy as np
from PIL import Image

class data():
    """docstring for data."""

    def __init__(self, data, labels, type):
        super(data, self).__init__()
        self.data = data
        self.lables = labels
        self.type = type


def readFile(dataFile, labelFile, type):
    dfile = open(dataFile)
    fileData = dfile.read()
    lfile = open(labelFile)
    labelData = lfile.read()
    d = data(fileData, labelData, type)
    lfile.close()
    dfile.close()
    return d

def preprocess(data, inputSize):
    type = data.type
    if (type=="image"):
        images = []
        images = data.data
        for i in images:
            images[i] = imageResize(images[i], inputSize)
    elif(type=="text"):
        raise NotImplementedError


    raise NotImplementedError

def imageResize(image):
    img = Image.open(image).convert('RGBA')
    arr = np.array(img)
    flatArr = arr.ravel()
    arrSize = len(flat_arr)
    flatArr /= flatArr
    return flatArr



def readFolder(dataFolder, labelFolder, type):
    raise NotImplementedError

def imageParse(data):
    raise NotImplementedError


class batch():
    """docstring for batch."""

    def __init__(self, data, type):
        super(batch, self).__init__()
        self.data = data
        self.type = type
        x = self.data.data
        y = self.data.lables


    def split(data, lables, percentData, type):


        training = data * percentData

        raise NotImplementedError

def load(data, lables, numbbaches): # split data into batches
    type = data.type;
    dataLength = len(data)
    segmentSize = datalength/numbacches
    batchs = {}
    for i in range(0, dataLength, segmentSize):
        batchs["batch{0}".format(i)] = batch(data[i:i+segmentSize], type)

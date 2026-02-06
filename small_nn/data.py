import os
import numpy as np
from PIL import Image


class data():
    """Data container for training data and labels."""

    def __init__(self, data, labels, type='numpy'):
        self.data = np.array(data, dtype=np.float64)
        self.labels = np.array(labels, dtype=np.float64)
        self.type = type
        self.batch_size = 32

    def batches(self):
        """Yield batches of (x, y) pairs."""
        n = len(self.data)
        for i in range(0, n, self.batch_size):
            yield self.data[i:i + self.batch_size], self.labels[i:i + self.batch_size]


def readFile(dataFile, labelFile, type='numpy'):
    """Read data and labels from files."""
    with open(dataFile) as dfile:
        fileData = dfile.read()
    with open(labelFile) as lfile:
        labelData = lfile.read()
    d = data(fileData, labelData, type)
    return d


def preprocess(dataset, inputSize):
    """Preprocess data based on its type."""
    if dataset.type == "image":
        images = dataset.data
        processed = []
        for img in images:
            processed.append(imageResize(img, inputSize))
        dataset.data = np.array(processed)
    elif dataset.type == "text":
        raise NotImplementedError("Text preprocessing not yet implemented")
    return dataset


def imageResize(image, inputSize=None):
    """Load an image, convert to array, and normalize."""
    img = Image.open(image).convert('RGBA')
    if inputSize is not None:
        img = img.resize((inputSize, inputSize))
    arr = np.array(img, dtype=np.float64)
    flatArr = arr.ravel()
    flatArr = flatArr / 255.0
    return flatArr


def readFolder(dataFolder, labelFolder, type='image'):
    """Read all files from data and label folders."""
    data_files = sorted(os.listdir(dataFolder))
    label_files = sorted(os.listdir(labelFolder))
    data_list = [os.path.join(dataFolder, f) for f in data_files]
    label_list = [os.path.join(labelFolder, f) for f in label_files]
    return data(data_list, label_list, type)


def imageParse(image_data):
    """Parse image data into a numpy array."""
    arr = np.array(image_data, dtype=np.float64)
    return arr.ravel() / 255.0


class batch():
    """A batch of training data."""

    def __init__(self, x, y):
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)


def split(dataset, percent_train=0.8):
    """Split data into training and testing sets."""
    n = len(dataset.data)
    split_idx = int(n * percent_train)
    indices = np.random.permutation(n)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    train_data = data(dataset.data[train_idx], dataset.labels[train_idx], dataset.type)
    test_data = data(dataset.data[test_idx], dataset.labels[test_idx], dataset.type)
    return train_data, test_data


def load(dataset, batch_size=32):
    """Split data into batches and return them."""
    dataset.batch_size = batch_size
    return dataset

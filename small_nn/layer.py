import numpy as np


class layer():
    """Base layer class for neural network layers."""

    def __init__(self, input_size, output_size, activation='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation


def dense(input_size, output_size, activation='relu'):
    """Create a fully connected (dense) layer."""
    return layer(input_size, output_size, activation)


def conv(input_channels, output_channels, kernel_size=3, activation='relu'):
    """Create a convolutional layer."""
    l = layer(input_channels, output_channels, activation)
    l.kernel_size = kernel_size
    l.type = 'conv'
    return l


def input_layer(size):
    """Create an input layer with the given size."""
    return layer(size, size, activation='identity')


def output_layer(input_size, output_size, activation='logistic'):
    """Create an output layer."""
    return layer(input_size, output_size, activation)

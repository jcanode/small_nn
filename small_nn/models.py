import numpy as np
from small_nn.Network import network


class model(object):
    """A model that wraps a network with layer definitions."""

    def __init__(self, optimizer='adam', learning_rate=0.001):
        self.layer_sizes = []
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.net = None

    def add(self, layer_size):
        """Add a layer size to the model."""
        self.layer_sizes.append(layer_size)

    def build(self):
        """Build the network from the added layers."""
        self.net = network(
            layers=self.layer_sizes,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
        )
        self.net.build()
        return self.net

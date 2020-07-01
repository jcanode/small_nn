import numpy as np
### # TODO: Connections
# # TODO: inputs/outputs
# TODO: training using cost function

class network(object):
    """docstring for network."""

    def __init__(self, arg):
        super(network, self).__init__()
        self.arg = arg


# def network():

# network = lambda node :
def activation(n): #relu max(0,x)
    if (n<=0):
        return 0;
    else:
        return n;

node = lambda n=0: activation(n)


layer = lambda n: [node for i in range(n)]

print (layer(3))

print(node(5))


def cost(target, actual): # mean square error
    mse = 0
    n = 1
    for i in actual:
        loss = (target[i] - actual[i])^2
        mse = mse + loss
        n = n+1
    mse = 1/n * mse
    return mse

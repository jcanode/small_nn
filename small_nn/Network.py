import numpy as np
###     # TODO: Connections
# # TODO: inputs/outputs
# TODO: training using cost function

class Network(object):
    """docstring for network."""

    def __init__(self, arg):
        super(network, self).__init__()
        self.arg = arg


def activation(n, func):
    if (func == "relu"):
        a = relu(n)
        return a;
    elif (func == "identity"):
        a = identity(n)
        return a;
    elif (func == "binary"):
        a = Binary(n)
        return a;
    elif (func == "logistic"):
        a = Sigmoid(n)
        return a;
    elif (func == "lrelu"):
        a = leakyRelu(n);
        return a;
    elif (func == "sine"):
        a = sine(n);
        return a;
    elif (func == "softmax"):
        a = softmax(n);
        return a;


"""Activation functions"""
def softmax(n):
    a = np.exp(n)/np.sum(np.exp(a))
    return a;

def sine(n):
    r = np.sin(n);
    return r;

def leakyRelu(n):
    if (n<0):
        return 0.01*n;
    else:
        return n;

def Sigmoid(n):
    r = 1.0/(1.0+np.exp(-n))
    return r;

def Binary(n):
    if (n < 0):
        return 0;
    else:
        return 1;

def identity(n):
    return n;

def relu(n):
    if (n<=0):
        return 0;
    else:
        return n;

"""End of Activation functions"""

node = lambda n=0: activation(n)


layer = lambda n: [node for i in range(n)]


def cost(target, actual): # mean square error
    mse = 0
    n = 1
    for i in actual:
        loss = (target[i] - actual[i])^2
        mse = mse + loss
        n = n+1
    mse = 1/n * mse
    return mse

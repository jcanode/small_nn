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

def train_step(model, data, t):
        lables = data.lables
        prediction = model.outputs
        loss = cost(lables,prediction)
        t = t+1
        optmizer(model, data, loss, t);
        return model, loss;

def opmizer(model, data, loss,t): # Defult adom
#params[weights,bias], vs, sqrs, learning_rate, batch_size, t
    alpha = 0.001 # step size
    beta1 = 0.9 # decay rate
    beta2=0.999
    epsilon=(10.0e-8)
    # theta = intial
    M0 = [0]    #initalize 1st vector
    M1 = [0]    #initalize 2nd vector
        for param, v, sqr in zip(params,vs,sqrs):
                g = Gradient_theta / data.batch_size # get gradient wrt stoacstic objective at t
                v[:] = beta1* v (1.0-beta1) * g # update first movement estimetn
                sqr[:] = beta2 * sqr  + (1.0-beta2) * (g ** 2) # vt = (1-beta2) * for i in range t:
                v_bias_corr = v/(1.0-beta1^t)
                sqr_bias_coor = sqr/(1.0-beta2^t)   #compute bias
                Theta_t = alpha * (sqr/(sqrt(sqr_bias_coor)+epsilon)) # update parameters
        return Theta_t


def vt(t):
    s = 0
    for i in range(t):
        n = np.dot(beta2^(t-i),g[i]^2)
        s=s+n;
    vt = (1-beta2) *s
    return vt;


def train(epochs, model, data):
    t = 0
    for epochs in range():
        model, loss = train_step(model, data, t)
        print(loss)
    return model;

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

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
    return {
        "relu": relu,
        "identity": identity,
        "binary": Binary,
        "logistic": Sigmoid,
        "lrelu": leakyRelu,
        "sine": sine,
        "softmax": softmax
    }.get(func, relu)(n) # default activation is relu


"""Activation functions"""
def softmax(n):
    return np.exp(n) / np.sum(np.exp(a))

def sine(n):
    return np.sin(n)

def leakyRelu(n):
    return (0.01 if n < 0 else 1) * n

def Sigmoid(n):
    return 1.0 / (1.0 + np.exp(-n))

def Binary(n):
    return 0 if n < 0 else 1
    if (n < 0):
        return 0
    else:
        return 1

def identity(n):
    return n

def relu(n):
    return 0 if n <= 0 else n

"""End of Activation functions"""

def train_step(model, data, t):
        lables = data.lables
        prediction = model.outputs
        loss = cost(lables,prediction)
        t = t+1
        optmizer(model, data, loss);
        return model, loss;

def optimize(model, data, loss): # Defult adom
#params[weights,bias], vs, sqrs, learning_rate, batch_size, t
    alpha = 0.001 # step size/learning_rate
    beta1 = 0.9 # decay rate
    beta2=0.999
    epsilon=(10.0e-8)
    # theta = intial
    _m = 0
    _v = 0
    M0 = [0]    #initalize 1st vector
    M1 = [0]    #initalize 2nd vector
    # t = 0       #initalize timestamp
        for param, v, sqr in zip(params,vs,sqrs):
                _m += (1.0 - beta1) * (g - _m)
                _v += (1.0 - beta2) * (g**2 - _v)
                _m = _m/(1/beta1**t)
                _v = _v/(1-beta2**t)
                Theta_t = -alpha *_m / (np.sqrt(_v)+epsilon)


                # g = Gradient_theta / data.batch_size # get gradient wrt stoacstic objective at t
                # v[:] = beta1 * v + (1.0-beta1) * g # update first movement estimetn
                # sqr[:] = beta2 * sqr  + (1.0-beta2) * (g ** 2) # vt = (1-beta2) * for i in range t:
                # v_bias_corr = v/(1.0-beta1^t)
                # sqr_bias_coor = sqr/(1.0-beta2^t)   #compute bias
                # Theta_t = alpha * (sqr/(sqrt(sqr_bias_coor)+epsilon)) # update parameters
        return Theta_t

def SGD(model, data):
    learning_rate = 0.001
    epochLoss = []
    for batch in data.batches:
        preds = Sigmoid(np.dot(data.batch.x, w))
        error = preds-data.batch.y
        loss = cost(preds, data.batch.y)
        epochLoss.append(loss)
        gradient =  np.dot(np.transpose(data.batch.x), error) / data.batch.x.shape[]
        W += learning_rate * gradient


def vt(t):
    s = 0
    for i in range(t):
        n = np.dot(beta2^(t-i),g[i]^2)
        s=s+n;
    vt = (1-beta2) *s
    return vt;


def train(epochs, model, data):
    for epoch in range(epochs):
        loss = 0
        # model, loss = train_step(model, data)
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

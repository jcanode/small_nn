import numpy as np


class network(object):
    """A simple neural network."""

    def __init__(self, layers=None, optimizer='adam', learning_rate=0.001):
        self.layers = layers or []
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

    def add(self, layer):
        self.layers.append(layer)

    def build(self):
        """Initialize weights and biases for each layer pair."""
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            w = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2.0 / self.layers[i])
            b = np.zeros((1, self.layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """Forward pass through the network."""
        self.activations = [X]
        self.z_values = []
        a = X
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = relu_array(z)
            self.activations.append(a)
        # Output layer uses sigmoid
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = sigmoid_array(z)
        self.activations.append(a)
        return a

    def backward(self, y):
        """Backpropagation to compute gradients."""
        m = y.shape[0]
        gradients_w = []
        gradients_b = []

        # Output layer error
        delta = self.activations[-1] - y

        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            gradients_w.insert(0, dw)
            gradients_b.insert(0, db)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * relu_derivative(self.z_values[i - 1])

        return gradients_w, gradients_b


def activation(n, func):
    return {
        "relu": relu,
        "identity": identity,
        "binary": Binary,
        "logistic": Sigmoid,
        "lrelu": leakyRelu,
        "sine": sine,
        "softmax": softmax
    }.get(func, relu)(n)


"""Activation functions"""


def softmax(n):
    exp_n = np.exp(n - np.max(n))
    return exp_n / np.sum(exp_n)


def sine(n):
    return np.sin(n)


def leakyRelu(n):
    return np.where(n < 0, 0.01 * n, n)


def Sigmoid(n):
    return 1.0 / (1.0 + np.exp(-n))


def sigmoid_array(n):
    return 1.0 / (1.0 + np.exp(-n))


def sigmoid_derivative(n):
    s = sigmoid_array(n)
    return s * (1 - s)


def Binary(n):
    return 0 if n < 0 else 1


def identity(n):
    return n


def relu(n):
    return 0 if n <= 0 else n


def relu_array(n):
    return np.maximum(0, n)


def relu_derivative(n):
    return (n > 0).astype(float)


"""End of Activation functions"""


def train_step(model, X, y):
    predictions = model.forward(X)
    loss = cost(y, predictions)
    gradients_w, gradients_b = model.backward(y)
    return gradients_w, gradients_b, loss


def optimize(model, X, y, optimizer='adam', t=1, m_w=None, v_w=None, m_b=None, v_b=None):
    """Adam optimizer."""
    alpha = model.learning_rate
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    gradients_w, gradients_b, loss = train_step(model, X, y)

    if m_w is None:
        m_w = [np.zeros_like(w) for w in model.weights]
        v_w = [np.zeros_like(w) for w in model.weights]
        m_b = [np.zeros_like(b) for b in model.biases]
        v_b = [np.zeros_like(b) for b in model.biases]

    for i in range(len(model.weights)):
        m_w[i] = beta1 * m_w[i] + (1.0 - beta1) * gradients_w[i]
        v_w[i] = beta2 * v_w[i] + (1.0 - beta2) * (gradients_w[i] ** 2)
        m_hat_w = m_w[i] / (1.0 - beta1 ** t)
        v_hat_w = v_w[i] / (1.0 - beta2 ** t)
        model.weights[i] -= alpha * m_hat_w / (np.sqrt(v_hat_w) + epsilon)

        m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * gradients_b[i]
        v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * (gradients_b[i] ** 2)
        m_hat_b = m_b[i] / (1.0 - beta1 ** t)
        v_hat_b = v_b[i] / (1.0 - beta2 ** t)
        model.biases[i] -= alpha * m_hat_b / (np.sqrt(v_hat_b) + epsilon)

    return loss, m_w, v_w, m_b, v_b


def SGD(model, X, y):
    """Stochastic Gradient Descent optimizer."""
    gradients_w, gradients_b, loss = train_step(model, X, y)
    for i in range(len(model.weights)):
        model.weights[i] -= model.learning_rate * gradients_w[i]
        model.biases[i] -= model.learning_rate * gradients_b[i]
    return loss


def train(epochs, model, data):
    """Train the network for a given number of epochs."""
    model.build()
    m_w, v_w, m_b, v_b = None, None, None, None
    t = 1
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in data.batches():
            if model.optimizer == 'sgd':
                loss = SGD(model, batch_x, batch_y)
            else:
                loss, m_w, v_w, m_b, v_b = optimize(
                    model, batch_x, batch_y, t=t,
                    m_w=m_w, v_w=v_w, m_b=m_b, v_b=v_b
                )
            epoch_loss += loss
            t += 1
        print("Epoch {}: loss = {:.6f}".format(epoch + 1, epoch_loss))
    return model


def cost(target, actual):
    """Mean squared error loss."""
    return np.mean((target - actual) ** 2)

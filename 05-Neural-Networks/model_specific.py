from tkinter.constants import X
from matplotlib.animation import AVConvFileWriter
import numpy as np
from numpy import linalg as LA

GaussianBasisSigma = 0.5

def groundTruthProcess(x, bias = 0.0, func = 'sin(2pix)'):
    """Computes underlying function 'func' we wish to discover.
 
    @param x N-dimentional vector of input data.
    @param bias bias of the function.
    @param func true function itself.

    @return N-dimentional target vector of function values.
    """

    if func == 'sin(2pix)':
        return bias + np.sin(2*np.pi*x)
    elif func == 'exp(x)':
        return bias + np.exp(x)
    elif func == 'x^2':
        return bias + x**2
    elif func == '|x|':
        return bias + np.abs(x)
    elif func == 'H(x)':
        return bias + np.heaviside(x, 0.5)
    
    return bias + np.sin(2*np.pi*x)

def generateData(N, low = -1.0, high = 1.0, func = 'sin(2pix)', bias = 0.0, noise_level = 0.3):
    """Generates data from some stochastic process (governed by "groundTruthProcess" function) with gaussian noise.
 
    @param N desierd number of data points.
    @param low low boundary of data points.
    @param high high boundary of data points.
    @param bias bias of the function.
    @param noise_level noise level of the function.

    @return N-dimentional target vector of stochastic process values.
    """

    x = np.random.uniform(low, high, size = N)
    t = groundTruthProcess(x, bias, func) + np.random.normal(0, noise_level, size = N)
    return x, t

def trainTestSplit(X, t, test_rate = 0.2):
    N = X.shape[0]
    rand_idx = np.random.choice(N, N, replace = False)
    valid_idx = rand_idx[:int(N * test_rate)]
    train_idx = rand_idx[int(N * test_rate):]
    return X[train_idx], t[train_idx], X[valid_idx], t[valid_idx]


class SimpleFeedForwardNetwork():
    """Simple two-layer (hidden + output) feed-forward neural network
    performing regression for one target variable.
    """

    def __init__(self) -> None:
        self.inputDim = 2 # including bias!
        self.NInputUnits = 2
        self.NHiddenUnits = 5
        self.NOutputUnits = 1
        self.w1 = np.random.normal(0.0, 0.2, size = (self.inputDim, self.NHiddenUnits))
        self.w2 = np.random.normal(0.0, 0.2, size = (self.NHiddenUnits, self.NOutputUnits))

    def fit(self, X, t, epochs = 100, batch_size = 32, learning_rate = 0.1, verbose = False):
        """Trains network using SGD.

        @param X NxD feature matrix of input data.
        @param t N-dimentional target vector of input data.
        @param epochs epoch number.
        @param batch_size batch number.
        @param learning_rate learning rate for SGD.
        @param verbose do print detailed info.
        """

        verboseInterval = epochs // 20
        if verboseInterval == 0:
            verboseInterval = 1

        N = X.size // self.inputDim
        for e in range(epochs):

            X_train, t_train, X_valid, t_valid = trainTestSplit(X, t)
            y_pred = np.zeros_like(t_train)
            for n, (x, tn) in enumerate(zip(X_train, t_train)):
                y, z = self.forward(x)
                y_pred[n] = y
                w1_grad, w2_grad = self.backward(y, tn, z, x)
                self.w1 -= learning_rate * w1_grad
                self.w2 -= learning_rate * w2_grad

            # rand_idx = np.random.choice(N, N, replace = False)
            # train_idx = rand_idx[:int(N * 0.8)]
            # valid_idx = rand_idx[int(N * 0.8):]
            # y_pred = np.zeros_like(rand_idx)
            # t_pred = np.take(t, train_idx)

            # iterNum = N // batch_size
            # w1_grad = np.zeros_like(self.w1)
            # w2_grad = np.zeros_like(self.w2)    
            # for i in range(iterNum):
            #     batch_rand_idx = rand_idx[i * batch_size : (i + 1) * batch_size]
            #     for n in batch_rand_idx:
            #         y, z = self.forward(X[n])
            #         y_pred[n] = y
            #         w1_grad_now, w2_grad_now = self.backward(y, t[n], z, X[n])
            #         w1_grad += w1_grad_now
            #         w2_grad += w2_grad_now
            #     self.w1 -= learning_rate * w1_grad
            #     self.w2 -= learning_rate * w2_grad

            # for n in train_idx:
            #     y, z = self.forward(X[n])
            #     y_pred[n] = y
            #     w1_grad, w2_grad = self.backward(y, t[n], z, X[n])
            #     self.w1 -= learning_rate * w1_grad
            #     self.w2 -= learning_rate * w2_grad

            y_valid_pred = np.zeros_like(t_valid)
            for n, (x, tn) in enumerate(zip(X_valid, t_valid)):
                y, _ = self.forward(x)
                y_valid_pred[n] = y

            if verbose and e % verboseInterval == 0:
                loss = self.RMS(y_pred, t_train)
                valid_loss = self.RMS(y_valid_pred, t_valid)
                print(">>> epoch {0}".format(e), "train loss (RMS): {0}".format(loss), "valid loss (RMS): {0}".format(valid_loss), sep = ', ')

    def forward(self, x):
        """Computes forward propagation.

        @param x D-dimentional vector of input data.

        @return Network output.
        """
        
        a = np.zeros(self.NHiddenUnits, dtype = np.float32)
        for j in range(self.NHiddenUnits):
            for i in range(self.NInputUnits):
                a[j] += self.w1[i, j] * x[i]

        z = np.tanh(a)

        y = np.zeros(self.NOutputUnits, dtype = np.float32)
        for k in range(self.NOutputUnits):
            for j in range(self.NHiddenUnits):
                y[k] += self.w2[j, k] * z[j]
        return y, z

    def backward(self, y, t, z, x):
        """Computes backward propagation.

        @param y network output.
        @param t ground-truth target.
        @param z hidden activation output.
        @param x network input.

        @return Weight gradients.
        """

        delta_output = (y - t)

        delta_hidden = np.zeros(self.NHiddenUnits, dtype = np.float32)
        for j in range(self.NHiddenUnits):
            for k in range(self.NOutputUnits):
                delta_hidden[j] += self.w2[j, k] * delta_output
        delta_hidden *= (1 - z**2)

        w1_grad = np.zeros_like(self.w1)
        for j in range(w1_grad.shape[1]):
            for i in range(w1_grad.shape[0]):
                w1_grad[i, j] = delta_hidden[j] * x[i]

        w2_grad = np.zeros_like(self.w2)
        for k in range(w2_grad.shape[1]):
            for j in range(w2_grad.shape[0]):
                w2_grad[j, k] = delta_output[k] * z[j]

        # w1_grad = np.outer(x, delta_hidden)
        # w2_grad = np.outer(z, delta_output)
        return w1_grad, w2_grad

    def loss(self, y, t):
        """Computes sum-of-squares loss function.

        @param y network output.
        @param t ground-truth target.

        @return loss.
        """

        return np.sum( (y - t)**2 ) / 2

    def RMS(self, y, t):
        """Computes root-mean-square function.

        @param y network output.
        @param t ground-truth target.

        @return loss.
        """

        N = t.size
        return np.sqrt( (2 * self.loss(y, t)) / N )
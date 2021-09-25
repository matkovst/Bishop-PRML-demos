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
    elif func == 'x+0.3sin(2pix)':
        return x + 0.3 * np.sin(2*np.pi*x)
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

    def __init__(self, NHiddenUnits) -> None:
        biasUnit = 1
        self.inputDim = 1 + biasUnit
        self.NInputUnits = 1 + biasUnit
        self.NHiddenUnits = NHiddenUnits + biasUnit
        self.NOutputUnits = 1
        self.w1 = np.random.normal(0.0, 0.2, size = (self.inputDim, self.NHiddenUnits))
        self.w2 = np.random.normal(0.0, 0.2, size = (self.NHiddenUnits, self.NOutputUnits))
        self.totalW = self.w1.size + self.w2.size
        self.W = np.zeros(self.totalW, dtype = np.float32)
        self.A = np.zeros((self.totalW, self.totalW), dtype = np.float32)
        self.al = 0.0
        self.bt = 1.0
        self.lmbd = 0.0

    def fit(self, X, t, epochs = 100, batch_size = 32, learning_rate = 0.1, lmbd = 0.0, al = 0.0, bt = 1.0, verbose = False):
        """Trains network using SGD.

        @param X NxD feature matrix of input data.
        @param t N-dimentional target vector of input data.
        @param epochs epoch number.
        @param batch_size batch number.
        @param learning_rate learning rate for SGD.
        @param lmbd regularization coefficient.
        @param al precision (inverse variance) of the prior weight distribution.
        @param bt precision (inverse variance) of input data.
        @param verbose do print detailed info.
        """

        self.al = al
        self.bt = bt
        self.lmbd = lmbd

        verboseInterval = epochs // 20
        if verboseInterval == 0:
            verboseInterval = 1

        N = X.size // self.inputDim
        X_train, t_train, X_valid, t_valid = trainTestSplit(X, t)
        for e in range(epochs):
            y_pred = np.zeros_like(t_train)
            for n, (x, tn) in enumerate(zip(X_train, t_train)):
                y, z = self.forward(x)
                y_pred[n] = y
                w1_grad, w2_grad = self.backward(y, tn, z, x)
                self.w1 = self.w1 - learning_rate * (w1_grad + self.lmbd * self.w1)
                self.w2 = self.w2 - learning_rate * (w2_grad + self.lmbd * self.w2)

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

        # Combine all weights into one vector for convenience
        self.W = np.concatenate((self.w1.flatten(), self.w2.flatten()))

        # Evaluate posterior distribution parameters
        _, self.A = self.wPosteriorParams(X, t)

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

    def calculateHessian(self, X, t):
        """Evaluates Hessian by Outer Product Approximation.
        
        @param X NxD feature matrix of input data.
        @param t N-dimentional target vector of input data.

        @return Hessian matrix.
        """

        Hessian = np.zeros((self.totalW, self.totalW), dtype = np.float32)

        for n, (x, tn) in enumerate(zip(X, t)):

            # Get ∇y
            y, z = self.forward(x)
            delta_hidden = np.zeros(self.NHiddenUnits, dtype = np.float32)
            for j in range(self.NHiddenUnits):
                for k in range(self.NOutputUnits):
                    delta_hidden[j] += self.w2[j, k]
            delta_hidden *= (1 - z**2)
            w1_grad = np.zeros_like(self.w1)
            for j in range(w1_grad.shape[1]):
                for i in range(w1_grad.shape[0]):
                    w1_grad[i, j] = delta_hidden[j] * x[i]
            w2_grad = np.zeros_like(self.w2)
            for k in range(w2_grad.shape[1]):
                for j in range(w2_grad.shape[0]):
                    w2_grad[j, k] = z[j]

            # Approximate Hessian
            b = np.concatenate((w1_grad.flatten(), w2_grad.flatten()))
            Hessian += np.outer(b, b.T)
        
        return Hessian

    def wPosteriorParams(self, X, t, al = 0.0, bt = 0.0):
        """Evaluates p(w|α) parameters by local Gaussian approximation.
        
        @param X NxD feature matrix of input data.
        @param t N-dimentional target vector of input data.
        @param al precision (inverse variance) of the prior weight distribution.
        @param bt precision (inverse variance) of input data.

        @return p(w|α) parameters.
        """

        Hessian = self.calculateHessian(X, t)

        al = self.al if al == 0.0 else al
        bt = self.bt if bt == 0.0 else bt
        A = al * np.identity(self.totalW) + bt * Hessian

        return Hessian, A

    def hyperparameterOptimization_experimental(self, X, t, a0, b0):

        N = t.size
        a = a0
        b = b0

        eps = 10**-2
        doStop = False
        aFound = False
        bFound = False
        itr = 0
        maxIter = 100
        while not doStop and itr < maxIter:
            a_prev = a
            b_prev = b
            lmbd = a / b

            # Calculate γ
            H, A = self.wPosteriorParams(X, t)
            eig_w, _ = LA.eig( b * H )

            # Update α
            gamma = np.sum( eig_w / (a + eig_w) )
            a = gamma / (self.W.T @ self.W)

            # Update β
            sum_of_sq = 0.0
            for i in range(N):
                sum_of_sq += (t[i] - float(self.forward(X[i])[0]))**2
            b = 1 / ( (1 / (N - gamma)) * sum_of_sq )

            dff1 = np.abs(a - a_prev)
            dff2 = np.abs(b - b_prev)
            if dff1 < eps:
                aFound = True
            if dff2 < eps:
                bFound = True
            if aFound and bFound:
                doStop = True
            itr += 1

        return a, b

    def predictBayesian(self, x):
        """Computes predictive distribution p(t|x, D, α, β).

        @param x new unobserved data point.

        @return Estimated mean and sigma of the predictive distribution
        """

        y, z = self.forward(x)

        delta_hidden = np.zeros(self.NHiddenUnits, dtype = np.float32)
        for j in range(self.NHiddenUnits):
            for k in range(self.NOutputUnits):
                delta_hidden[j] += self.w2[j, k]
        delta_hidden *= (1 - z**2)

        w1_grad = np.zeros_like(self.w1)
        for j in range(w1_grad.shape[1]):
            for i in range(w1_grad.shape[0]):
                w1_grad[i, j] = delta_hidden[j] * x[i]

        w2_grad = np.zeros_like(self.w2)
        for k in range(w2_grad.shape[1]):
            for j in range(w2_grad.shape[0]):
                w2_grad[j, k] = z[j]

        g = np.concatenate((w1_grad.flatten(), w2_grad.flatten()))
        sigma2 = 0.0000001
        if LA.det(self.A) > 0:
            A_inv = LA.inv(self.A)
            sigma2 = 1/self.bt + g.T @ A_inv @ g

        return y, np.sqrt(sigma2)

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
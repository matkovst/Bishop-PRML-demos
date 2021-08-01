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

def Basis(x, i, mu = 0.0, sigma = 1.0, basis = 'power'):
    """Computes basis function for input data.
 
    @param x N-dimentional vector of input data.
    @param i index of basis function in polynom.
    @param mu (used only with 'gaussian' or 'sigmoid' basis) mean of gaussian basis.
    @param sigma (used only with 'gaussian' or 'sigmoid' basis) stdev of gaussian basis.

    @return N-dimentional vector of basis function values.
    """

    if basis == 'power':
        return np.power(x, i + 1)
    elif basis == 'gaussian':
        return np.exp(-(x - mu)**2 / (2*sigma**2))
    elif basis == 'sigmoid':
        return 1 / (1 + np.exp( -(x - mu) / sigma ))

def polynom(x, w, w0 = 0.0, basis = 'power'):
    """Computes polynom function with given basis.
 
    @param x N-dimentional vector of input data.
    @param w M-dimentional vector of parameters.
    @param w0 bias parameter.
    @param basis name of basis functions.

    @return polynom value.
    """

    if x.size == 1:
        x = np.array([x])

    M = w.size
    y = np.zeros(x.shape[0], dtype = np.float32)

    y += w0
    for i in range(M):
        mu = i / M
        sigma = GaussianBasisSigma
        y += w[i] * Basis(x, i, mu = mu, sigma = sigma, basis = basis)
    return y

def makeFeatureMatrix(x, M, basis = 'power'):
    """Creates feature matrix from input data according to polynomial function.
 
    @param x N-dimentional vector of input data.
    @param M order of polynom.
    @param basis name of basis functions.

    @return NxM feature matrix.
    """

    N = x.shape[0]
    P = np.zeros((N, M), dtype = np.float32)
    for i in range(M):
        mu = i / M
        sigma = GaussianBasisSigma
        P[:, i] = Basis(x, i, mu = mu, sigma = sigma, basis = basis)
    return P

def fitEntireSet(X, t, lmbd = 0.01):
    """Solves linear regression problem for given data in one go.
 
    @param X NxM feature matrix of input data.
    @param t N-dimentional target vector of input data.
    @param lmbd regularization coefficient.

    @return Estimated parameters w, w0, β.
    """

    ## --- Yet another way to solve w_est (unfortunately without regularization)

    # MoorePenrose = LA.pinv(X)
    # w_est = MoorePenrose @ t
    # return w_est

    ## ---

    N = X.shape[0]
    M = X.shape[1]
    RegTerm = lmbd * np.identity(M, dtype = np.float32)

    # w
    Matx = RegTerm + X.T @ X
    w_est = (LA.inv(Matx) @ X.T) @ t

    # w0
    t_mean = np.mean(t)
    phi_mean = np.sum(np.mean(X, axis = 0) * w_est)
    w0_est = t_mean - phi_mean

    # precision β (inverse variance)
    variance_est = 0.0
    for i in range(N):
        variance_est += (t[i] - w_est.T @ X[i, :])**2
    variance_est /= N
    b_est = 1 / variance_est

    return w_est, w0_est, b_est

def fitSequentially(X, t, lmbd = 0.01, batch_size = 1):
    """Solves linear regression problem for given data in a batch manner via SGD.
 
    @param X NxM feature matrix of input data.
    @param t N-dimentional target vector of input data.
    @param lmbd regularization coefficient.
    @param batch_size batch number.

    @return Estimated parameters w, w0, β.
    """

    N = X.shape[0]
    M = X.shape[1]
    X_mean = np.mean(X, axis = 0)
    t_mean = np.mean(t)

    # Initialize params
    w_est = np.zeros(M, dtype = np.float32)
    w0_est = 0.0

    nu = 0.5
    iterNum = N // batch_size
    for i in range(iterNum):

        # w
        w_prev = w_est.copy()
        grad_E = 0.0
        for j in range(batch_size):
            grad_E += (t[i + j] - w_prev.T @ X[i + j, :]) * X[i + j, :]
        grad_E /= batch_size
        w_est = w_prev + nu * grad_E

        # w0
        phi_mean = np.sum(X_mean * w_est)
        w0_est = t_mean - phi_mean

    # precision β (inverse variance)
    variance_est = 0.0
    for i in range(N):
        variance_est += (t[i] - w_est.T @ X[i, :])**2
    variance_est /= N
    b_est = 1 / variance_est

    return w_est, w0_est, b_est

def wPosteriorParams(X, t, b, lmbd):
    """Computes mu and covar parameters of weight posterior normal distribution.

    @param X NxM feature matrix of input data.
    @param t N-dimentional target vector of input data.
    @param b precision (inverse variance) of input data
    @param lmbd regularization coefficient.

    @return Estimated mean and covar of the posterior distribution
    """

    M = X.shape[1]
    alpha = lmbd * b
    aI = alpha * np.identity(M, dtype = np.float32)
    Covar_inv = aI + (b * X.T) @ X
    Covar = LA.inv( Covar_inv )
    Mean = b * Covar @ X.T @ t
    return Mean, Covar

def predictiveDistrParams(x, b, MeanPoster, CovarPoster, basis = 'power'):
    """Computes mu and sigma parameters of target variable predictive normal distribution.

    @param x new unobserved data point.
    @param b precision (inverse variance) of input data
    @param MeanPoster mean of weight prior distribution.
    @param CovarPoster Covariance matrix of weight prior distribution.
    @param basis name of basis functions.

    @return Estimated mean and sigma of the predictive distribution
    """

    M = MeanPoster.size
    phi_x = np.zeros(M, dtype = np.float32)
    for i in range(M):
        mu = i / M
        sigma = GaussianBasisSigma
        phi_x[i] = Basis(x, i, mu = mu, sigma = sigma, basis = basis)

    Sigma2 = 1/b + phi_x.T @ CovarPoster @ phi_x
    Sigma = np.sqrt(Sigma2)
    Mean = MeanPoster.T @ phi_x

    return Mean, Sigma

def evaluateEvidence(X, t, a, b):
    """Evaluates the evidence function p(t|α,β) in order to find data preference for the model.
    Using this knowledge we can favour a model with highest preference.

    @param X NxM feature matrix of input data.
    @param t N-dimentional target vector of input data.
    @param a precision (inverse variance) of the prior weight distribution
    @param b precision (inverse variance) of input data

    @return Estimated mean and sigma of the predictive distribution
    """

    N = X.shape[0]
    M = X.shape[1]
    TwoPi = 2 * np.pi
    Const_1 = (b/TwoPi)**(N/2)
    Const_2 = (a/TwoPi)**(M/2)
    Const_3 = TwoPi**(M/2)
    aI = a * np.identity(M, dtype = np.float32)
    A = aI + b * X.T @ X
    mn = b * LA.inv(A) @ X.T @ t
    E_mn = (b/2) * np.sum((t - X @ mn)**2) + (a/2) * mn.T @ mn
    Evidence = Const_1 * Const_2 * np.exp(-E_mn) * Const_3 * (1/np.sqrt(LA.det(A)))
    return Evidence

def estimateHyperparams(X, t, a0, b0):
    """Estimates linear model hyperparameters α and β using Bayesian evidence approximation.

    @param X NxM feature matrix of input data.
    @param t N-dimentional target vector of input data.
    @param a precision (inverse variance) of the prior weight distribution
    @param b precision (inverse variance) of input data

    @return Estimated mean and sigma of the predictive distribution
    """

    N = t.size
    a = a0
    b = b0
    _, eig_v = LA.eig( b * X.T @ X )

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
        wMean, _ = wPosteriorParams(X, t, b, lmbd)
        _, eig_v = LA.eig( b * X.T @ X )

        gamma = np.sum( eig_v / (a + eig_v) )
        a = gamma / (wMean.T @ wMean)
        sum_of_sq = 0.0
        for i in range(N):
            sum_of_sq += (t[i] - wMean.T @ X[i, :])**2
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

def RMS(x, w, t, basis = 'power'):
    """Compute Root-mean-square error.
    
    @param x N-dimentional vector of input data.
    @param w M-dimentional vector of estimated parameters.
    @param t N-dimentional target vector of input data.
    @param basis name of basis functions.

    @return Estimated error.
    """
    
    N = t.size
    E_w = 0.0
    for i in range(N):
        E_w += (t[i] - polynom(x[i], w, basis = basis))**2
    E_rms = np.sqrt(E_w / N)
    return E_rms.item()
import sys
import numpy as np
from numpy import linalg as LA
from numpy.linalg.linalg import norm

NDim = 2
Stdev = 5

def generateBlobs(N_points, N_classes, distr = 'gaussian'):
    X = np.zeros((N_points, NDim), dtype = np.float32)
    y = np.zeros(N_points, dtype = np.int32)
    N_per_class = N_points // N_classes
    for c in range(N_classes):
        if distr.lower() == 'gaussian':
            Mean = [15 * c, 15 * (c % 2)]
            Covar = [[Stdev**2, Stdev * 7 * ((c + 1) % 2)], [-Stdev * 7 * ((c + 2) % 2), Stdev**2]]
            X[N_per_class * c : N_per_class * (c + 1)] = np.random.multivariate_normal(Mean, Covar, N_per_class)
            y[N_per_class * c : N_per_class * (c + 1)] = c
        elif distr.lower() == 'mog':
            ws = np.array([0.5, 0.5])
            Means = np.array([[0, 0], [50, 50]])
            Covars = np.array([[[Stdev**2, 3], [2, Stdev**2]], [[Stdev**2, 3], [2, Stdev**2]]])
            X[N_per_class * c : N_per_class * (c + 1)] = generateFromMOG(Means, Covars, ws, (N_per_class, NDim))
            y[N_per_class * c : N_per_class * (c + 1)] = c
    return X, y

def generateFromMOG(Mus, Covars, ws, size):
    N = size[0]
    R = np.zeros(size, dtype = np.float32)
    for m in range(Mus.shape[0]):
        rnd = np.random.uniform(Mus[m] - 3 * np.diag(Covars[m]), Mus[m] + 3 * np.diag(Covars[m]), size = size)
        R[:, 0] += ws[m] * GaussianPDF(rnd[:, 0], Mus[m, 0], Covars[m, 0, 0])
        R[:, 1] += ws[m] * GaussianPDF(rnd[:, 1], Mus[m, 1], Covars[m, 1, 1])
    return R

def GaussianPDF(x, mu, var):
    denom = (2 * np.pi * var)**.5
    num = np.exp(-(x - mu)**2 / (2 * var))
    return num / denom

def multiGaussianPDF(x, mu, covar):
    d = mu.size
    det = np.linalg.det(covar)
    if det <= 0:
        return 0.99
    normCoeff = (1. / (1 * np.pi)**(d/2)) * (1. / det**(1/2))
    covar_inv = np.linalg.inv(covar)
    dff = (x - mu)
    expArg = np.clip((dff @ covar_inv @ dff.T) / (-2), -709.78, 709.78) # <- clip for float64 to avoid "RuntimeWarning: overflow encountered in exp"
    Exp = np.exp( expArg )
    res = normCoeff * Exp
    if res == np.inf:
        return 1.0
    elif res == -np.inf:
        return 0.0
    return normCoeff * Exp

def standardizing(data):
    """Rescales input data as (data - mu) / sigma.
    
    @param data NxD input data.

    @return Rescaled data.
    """

    means = np.mean(data, axis = 0)
    stdevs = np.std(data, axis = 0)
    return (data - means) / stdevs


class Classificator:
    def fit(self):
        NotImplementedError("fit() method must be overriden")

    def evaluate(self):
        NotImplementedError("evaluate() method must be overriden")

    def predict(self, x):
        NotImplementedError("predict() method must be overriden")


class KMeansClassificator(Classificator):
    def __init__(self) -> None:
        self._Means = None
        self._R = None
        self._Jhistory = None

    def fit(self, X, K, epochs = 5):
        """Solves clustering problem with k-means for given data in one go.
    
        @param X NxD feature matrix of input data.
        @param K number of desired clusters.
        @param epochs how many epochs it should take.

        @return Estimated cluster means.
        """

        self._Xtrain = X.copy()
        dims = X.shape[1]
        self._N = X.shape[0]
        self._Means = np.random.normal(size = (K, dims))
        self._R = np.zeros((self._N, K), dtype = np.float32)

        self._Jhistory = np.zeros(epochs * 2, dtype = np.float32)
        for e in range(epochs):
            # E-step
            self._R.fill(0)
            for n, xn in enumerate(X):
                dists = np.linalg.norm(xn - self._Means, axis = 1)**2
                self._R[n, np.argmin(dists)] = 1
            
            self._Jhistory[e*2] = self.__distortion(X, self._R, self._Means)

            # M-step
            for k in range(K):
                self._Means[k] = np.sum(X[self._R[:, k] == 1, :], axis = 0) / np.sum(self._R[:, k])

            self._Jhistory[e*2 + 1] = self.__distortion(X, self._R, self._Means)

        return self._Means

    def fitSequential(self, X, K = 2, learning_rate = 0.05):
        """Solves clustering problem with k-means for given data sequentially.
    
        @param X NxD feature matrix of input data.
        @param K number of desired clusters.
        @param learning_rate learning rate for update formula.

        @return Estimated cluster means.
        """

        self._Xtrain = X.copy()
        dims = X.shape[1]
        self._N = X.shape[0]
        self._Means = np.random.normal(size = (K, dims))
        self._R = np.zeros((self._N, K), dtype = np.float32)

        for n, xn in enumerate(X):
            dists = np.linalg.norm(xn - self._Means, axis = 1)**2
            k = np.argmin(dists)
            self._R[n, k] = 1
            self._Means[k] += learning_rate * (xn - self._Means[k])

        return self._Means

    def evaluate(self, t):
        """Evaluates model accuracy given true clusters.

        @param t N-dimentional target vector of input data.

        @return Estimated accuracy
        """

        miscl = 0
        for n, xn in enumerate(self._Xtrain):
            if self.predict(xn) != t[n]:
                miscl += 1
        return 100 - (miscl / t.size) * 100

    def predict(self, x):
        """Predicts cluster for given points.

        @param x D-dimentional point or NxD matrix of points

        @return Predicted cluster
        """

        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        N = x.shape[0]
        preds = np.zeros(N)
        for n, xn in enumerate(x):
            dists = np.linalg.norm(xn - self._Means, axis = 1)**2
            preds[n] = np.argmin(dists)
        return preds

    def Means(self):
        """Returns estimated cluster means.

        @return means
        """

        return self._Means

    def __distortion(self, X, R, mus):
        """Measures distortion function.

        @param X NxD feature matrix of input data.
        @param R NxK responce matrix of input data.
        @param mus cluster means.

        @return distortion measure
        """

        J = 0.0
        for n, xn in enumerate(X):
            for k in range(mus.shape[0]):
                J += R[n, k] * np.linalg.norm(xn - mus[k])**2
        return J


class MOGClassificator(Classificator):
    def __init__(self) -> None:
        self._Means = None
        self._R = None
        self._Jhistory = None
        self.kMeans = KMeansClassificator() # <- for mean vector initialization

    def fit(self, X, K, epochs = 5):
        """Solves clustering problem with MOG for given data in one go.
    
        @param X NxD feature matrix of input data.
        @param K number of desired clusters.
        @param epochs how many epochs it should take.

        @return Estimated cluster means.
        """

        self._Xtrain = X.copy()
        dims = self._Xtrain.shape[1]
        self._N = self._Xtrain.shape[0]
        Ns = np.zeros(K)
        #self._Means = np.random.normal(size = (K, dims))
        self._Means = self.kMeans.fit(self._Xtrain, K, epochs)
        self._Covars = np.random.normal(size = (K, dims, dims))
        self._Ps = np.random.uniform(size = K)
        self._Ps /= np.sum(self._Ps)
        self._R = np.zeros((self._N, K), dtype = np.float32)

        self._LogLikl_history = np.zeros(epochs + 1, dtype = np.float32)
        self._LogLikl_history[0] = self.__logLikelihood(self._Xtrain, self._Means, self._Covars, self._Ps)
        for e in range(epochs):
            # E-step
            self._R.fill(0)
            for n, xn in enumerate(self._Xtrain):
                for k in range(K):
                    self._R[n, k] = self._Ps[k] * multiGaussianPDF(xn, self._Means[k], self._Covars[k])
                    if np.isnan(self._R[n, k]):
                        self._R[n, k] = 0.0
                self._R[n] /= np.sum(self._R[n])

            # M-step
            self._Means.fill(0)
            self._Covars.fill(0)
            self._Ps.fill(0)
            Ns = np.sum(self._R, axis = 0)
            for k in range(K):
                for n, xn in enumerate(self._Xtrain):
                    self._Means[k] += self._R[n, k] * xn
                self._Means[k] /= Ns[k]
            for k in range(K):
                for n, xn in enumerate(self._Xtrain):
                    self._Covars[k] += self._R[n, k] * np.outer((xn - self._Means[k]), (xn - self._Means[k]))
                self._Covars[k] /= Ns[k]
            for k in range(K):
                self._Ps[k] = Ns[k] / self._N

            self._LogLikl_history[e + 1] = self.__logLikelihood(self._Xtrain, self._Means, self._Covars, self._Ps)

        return self._Means

    def evaluate(self, t):
        """Evaluates model accuracy given true clusters.
        NOTE: The accuracy is identifiability-prone.

        @param t N-dimentional target vector of input data.

        @return Estimated accuracy
        """

        miscl = 0
        for n, xn in enumerate(self._Xtrain):
            if self.predict(xn) != t[n]:
                miscl += 1
        return 100 - (miscl / t.size) * 100

    def predict(self, x):
        """Performes "hard" clustering for given points.

        @param x D-dimentional point or NxD matrix of points

        @return Predicted cluster
        """

        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        N = x.shape[0]
        K = self._Means.shape[0]
        class_preds = np.zeros(N, dtype = np.int32)
        for n, xn in enumerate(x):
            proba = np.zeros(K, dtype = np.float32)
            for k in range(K):
                proba[k] = self._Ps[k] * multiGaussianPDF(xn, self._Means[k], self._Covars[k])
            proba /= np.sum(proba)
            class_preds[n] = np.argmax(proba)

        return class_preds

    def predict_proba(self, x):
        """Predicts nearest cluster probability for given points.

        @param x D-dimentional point or NxD matrix of points

        @return Predicted probability
        """

        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        N = x.shape[0]
        K = self._Means.shape[0]
        proba_preds = np.zeros((N, K), dtype = np.float32)
        for n, xn in enumerate(x):
            for k in range(K):
                proba_preds[n, k] = self._Ps[k] * multiGaussianPDF(xn, self._Means[k], self._Covars[k])
            proba_preds[n] /= np.sum(proba_preds[n])

        return proba_preds

    def Means(self):
        """Returns estimated cluster means.

        @return means
        """

        return self._Means

    def Covars(self):
        """Returns estimated cluster covariances.

        @return covariances
        """

        return self._Covars

    def R(self):
        """Returns estimated cluster responce matrix.

        @return responce matrix
        """

        return self._R

    def __logLikelihood(self, X, means, covars, ps):
        """Measures MOG log likelihood function.

        @param X NxD feature matrix of input data.
        @param means estimated means of input data.
        @param covars estimated covariances of input data.
        @param ps estimated mixing coefficients of input data.

        @return log likelihood measure
        """

        error = 0.0
        for n, xn in enumerate(X):
            arg = 0.0
            for k in range(means.shape[0]):
                arg += ps[k] * multiGaussianPDF(xn, means[k], covars[k])
            error += np.log(arg)
        return error
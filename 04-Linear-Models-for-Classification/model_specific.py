import sys
import numpy as np
from numpy import linalg as LA

NDim = 2
Stdev = 5

def generateBlobs(N_points, N_classes, distr = 'gaussian'):
    X = np.zeros((N_points, NDim), dtype = np.float32)
    y = np.zeros(N_points, dtype = np.int32)
    N_per_class = N_points // N_classes
    for c in range(N_classes):
        if distr.lower() == 'gaussian':
            Mean = [15 * c, 15 * (c % 2)]
            Covar = [[Stdev**2, Stdev], [Stdev, Stdev**2]]
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

def makeFeatureMatrix(x):
    X = np.ones((x.shape[0], NDim + 1), dtype = np.float32)
    X[:, 1:] = x
    return X

def lineEq(X, w):
    if X.ndim < 2:
        X = np.expand_dims(X, axis = 0)

    N = X.shape[0]
    K = w.shape[1] if w.ndim > 1 else 1
    R = np.zeros((N, K), dtype = np.float32)

    xLen = X.shape[1]
    wLen = w.shape[0] if w.ndim > 1 else w.size
    xHasBias = True if xLen == wLen else False
    if not xHasBias:
        X = makeFeatureMatrix(X)

    if N > 1:
        for i in range(N):
            R[i, :] = w.T @ X[i, :]
    else:
        R = w.T @ X[0, :]
    
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
    Exp = np.exp( (dff @ covar_inv @ dff.T) / (-2) )
    return normCoeff * Exp

def stepFunction(x):
    if x.ndim > 1 and x.shape[1] == 1:
        x = x[:, 0]
    N = x.shape[0] if x.ndim > 1 else x.size
    R = np.zeros(N, dtype = np.int32)
    R[np.where(x >= 0)] = +1
    R[np.where(x < 0)] = -1
    return R

def sigmoid(x):
    x_clipped = np.clip(x, -709.78, 709.78) # <- clip for float64 to avoid "RuntimeWarning: overflow encountered in exp"
    s = 1 / (1 + np.exp( -x_clipped, dtype = np.float32 ))
    if s.size == 1:
        s = s if s > 0 else 0.000001
        s = s if s < 1 else 0.999999
    else:
        s[np.where(s <= 0)] = 0.000001
        s[np.where(s >= 1)] = 0.999999
    return s

# def sigmoid(x):
#     return 1 / (1 + np.exp( -x ))

def softmax(x):
    if x.ndim > 1: # <- K-class case
        S = np.zeros_like(x)
        denom = np.sum(np.exp(x), axis = 1)
        for i in range(S.shape[1]):
            S[:, i] = np.exp(x[:, i]) / denom
        return S
    else: # <- Single-class case
        return np.exp(x) / np.sum(np.exp(x))


class Classificator:
    def fit(self):
        NotImplementedError("fit() method must be overriden")

    def evaluate(self):
        NotImplementedError("evaluate() method must be overriden")

    def predict(self, x):
        NotImplementedError("predict() method must be overriden")

class LeastSquaresClassificator(Classificator):
    def __init__(self, X, t) -> None:
        self._Xtrain = makeFeatureMatrix(X)
        self._Ttrain = self.__makeConvenientTargetVector(t)
        self._N = self._Xtrain.shape[0]
        self._K = np.unique(self._Ttrain).size
        self._W_est = np.zeros((NDim + 1, self._K), dtype = np.float32)

    def fit(self):
        Matx = self._Xtrain.T @ self._Xtrain
        self._W_est = (LA.inv(Matx) @ self._Xtrain.T) @ self._Ttrain
        return self._W_est

    def evaluate(self):
        Pred = self.predict(self._Xtrain)
        errors = 0
        for i in range(self._N):
            if Pred[i] != np.argmax(self._Ttrain[i]):
                errors += 1
        return 100 - (errors / self._N) * 100

    def predict(self, x):
        R = lineEq(x, self._W_est)
        return np.argmax(R, axis = 1)

    def __makeConvenientTargetVector(self, t):
        T = np.zeros((t.size, t.max() + 1))
        T[np.arange(t.size), t] = 1
        return T

class FisherClassificator(Classificator):
    def __init__(self, X, t) -> None:
        self._Xtrain = X
        self._Ttrain = t
        self._W_est = np.zeros(NDim + 1, dtype = np.float32)
        self._N = self._Xtrain.shape[0]
        self._K = np.unique(self._Ttrain).size

    def fit(self):
        # Calc means
        Nsamples = np.zeros(self._K, dtype = np.int32)
        Means = np.zeros((NDim, self._K), dtype = np.float32)
        for i in range(self._N):
            k = self._Ttrain[i]
            Nsamples[k] += 1
            Means[:, k] += self._Xtrain[i, :]
        Means /= Nsamples

        # Calc covars
        Covars = np.zeros((NDim, NDim, self._K), dtype = np.float32)
        for i in range(self._N):
            k = self._Ttrain[i]
            dff = self._Xtrain[i, :] - Means[:, k]
            Covars[:, :, k] += np.outer(dff, dff)

        # Two-classes
        Sw = Covars[:, :, 0] + Covars[:, :, 1]
        Sw_inv = LA.inv(Sw)
        W_est = Sw_inv @ (Means[:, 1] - Means[:, 0])

        # Estimate threshold (bias)
        y1 = []
        y2 = []
        for i in range(self._N):
            k = self._Ttrain[i]
            x = np.expand_dims(self._Xtrain[i, :], 0)
            if k == 0:
                y1.append(lineEq(x, W_est))
            else:
                y2.append(lineEq(x, W_est))
        y1 = np.array(y1)
        y2 = np.array(y2)
        y1_mean = np.mean(y1)
        y2_mean = np.mean(y2)
        y1_variance = np.sum( (y1 - y1_mean)**2 ) / (Nsamples[0] - 1)
        y2_variance = np.sum( (y2 - y2_mean)**2 ) / (Nsamples[1] - 1)
        A = 1/(2*y1_variance) - 1/(2*y2_variance)
        B = y2_mean/(y2_variance) - y1_mean/(y1_variance)
        C = y1_mean**2 /(2*y1_variance) - y2_mean**2 / (2*y2_variance) - np.log(y2_variance/y1_variance)
        roots = np.roots([A, B, C])
        w0 = -roots[0] if roots[0] >= 0 and roots[0] <= 1 else -roots[1]
        self._W_est = np.insert(W_est, 0, w0)

        return self._W_est

    def evaluate(self):
        Pred = self.predict(self._Xtrain)
        errors = 0
        for i in range(self._N):
            k = self._Ttrain[i]
            if Pred[i] != k:
                errors += 1
        return 100 - (errors / self._N) * 100

    def predict(self, x):
        N = x.shape[0] if x.ndim > 1 else x.size
        R = lineEq(x, self._W_est)
        t_pred = np.zeros(N, dtype = np.int32)
        for i in range(N):
            t_pred[i] = 0 if R[i] < 0 else 1
        return t_pred

class PerceptronClassificator(Classificator):
    def __init__(self, X, t) -> None:
        self._Xtrain = makeFeatureMatrix(X)
        self._Ttrain = self.__makeConvenientTargetVector(t)
        self._W_est = np.zeros(NDim + 1, dtype = np.float32)
        self._N = self._Xtrain.shape[0]

    def fit(self, learning_rate = 0.1):
        epochs = 100
        W_est = np.zeros(NDim + 1, dtype = np.float32)
        for e in range(epochs):
            grad_E = 0
            for i in range(self._N):
                y = stepFunction(W_est.T @ self._Xtrain[i, :])
                if y != self._Ttrain[i]:
                    grad_E += self._Xtrain[i, :] * self._Ttrain[i]
            W_est = W_est + learning_rate * grad_E

        self._W_est = W_est
        return self._W_est

    def evaluate(self):
        errors = 0
        Pred = self.predict(self._Xtrain)
        for i in range(self._N):
            k = 0 if self._Ttrain[i] == 1 else 1
            if Pred[i] != k:
                errors += 1
        return 100 - (errors / self._N) * 100

    def predict(self, x):
        N = x.shape[0] if x.ndim > 1 else x.size
        t_pred = stepFunction( lineEq(x, self._W_est) )
        # for i in range(N):
        #     t_pred[i] = stepFunction(self._W_est.T @ x[i, :])
        t_pred[np.where(t_pred == +1)] = 0
        t_pred[np.where(t_pred == -1)] = 1
        return t_pred

    def __makeConvenientTargetVector(self, t):
        t_new = np.zeros(t.size, dtype = np.int32)
        t_new[np.where(t == 0)] = +1
        t_new[np.where(t == 1)] = -1
        return t_new

class GenerativeGaussianClassificator(Classificator):
    def __init__(self, X, t) -> None:
        self._Xtrain = X
        self._Ttrain = t
        self._N = self._Xtrain.shape[0]
        self._K = np.unique(self._Ttrain).size
        self._W_est = np.zeros((NDim + 1, self._K), dtype = np.float32)
        
        self.Priors_ml = np.zeros(self._K, dtype = np.float32)
        self.Means_ml = np.zeros((NDim, self._K), dtype = np.float32)
        self.Covars_ml = np.zeros((NDim, NDim, self._K), dtype = np.float32)
        self.Covar_ml = np.zeros((NDim, NDim), dtype = np.float32)

    def fit(self):
        # Estimate priors
        for i in range(self._N):
            k = self._Ttrain[i]
            self.Priors_ml[k] += 1
        self.Priors_ml /= self._N

        # Estimate means
        for i in range(self._N):
            k = self._Ttrain[i]
            self.Means_ml[:, k] += self._Xtrain[i, :]
        self.Means_ml /= (self.Priors_ml * self._N)

        # Estimate covars
        for i in range(self._N):
            k = self._Ttrain[i]
            dff = self._Xtrain[i, :] - self.Means_ml[:, k]
            self.Covars_ml[:, :, k] += np.outer(dff, dff)
        self.Covars_ml /= (self.Priors_ml * self._N)

        # Calc shared covar
        self.Covar_ml = np.sum((self.Priors_ml * self._N) * (self.Covars_ml), axis = 2)

        # Estimate w
        Covar_ml_inv = LA.inv(self.Covar_ml)
        if self._K == 2: # <- Two-class case
            self._W_est[0, 0] = (-1 / 2) * self.Means_ml[:, 0].T @ Covar_ml_inv @ self.Means_ml[:, 0] + \
                (1 / 2) * self.Means_ml[:, 1].T @ Covar_ml_inv @ self.Means_ml[:, 1] + \
                np.log(self.Priors_ml[0] / self.Priors_ml[1])
            self._W_est[1:, 0] = Covar_ml_inv @ (self.Means_ml[:, 0] - self.Means_ml[:, 1])

        else: # <- K-class case
            for k in range(self._K):
                self._W_est[0, k] = (-1 / 2) * self.Means_ml[:, k].T @ Covar_ml_inv @ self.Means_ml[:, k] + \
                    np.log(self.Priors_ml[k])
                self._W_est[1:, k] = Covar_ml_inv @ self.Means_ml[:, k]

        return self._W_est

    def evaluate(self):
        errors = np.count_nonzero(self._Ttrain - self.predict(self._Xtrain))
        return 100 - (errors / self._N) * 100

    def predict(self, x):
        N = x.shape[0] if x.ndim > 1 else x.size
        t_pred = np.zeros(N, dtype = np.int32)
        if self._K == 2: # <- Two-class case
            for i in range(N):
                posterior1 = sigmoid(self._W_est[1:, 0].T @ x[i, :] + self._W_est[0, 0])
                posterior2 = 1 - posterior1
                t_pred[i] = 0 if posterior1 > posterior2 else 1

        else: # <- K-class case
            for i in range(N):
                a = self._W_est[1:, :].T @ x[i, :] + self._W_est[0, :]
                posteriors = softmax(a)
                t_pred[i] = np.argmax(posteriors)

        return t_pred

    def predict_proba(self, x):
        N = x.shape[0] if x.ndim > 1 else x.size
        probs = np.zeros((N, self._K), dtype = np.float32)
        if self._K == 2: # <- Two-class case
            for i in range(N):
                posterior1 = sigmoid(self._W_est[1:, 0].T @ x[i, :] + self._W_est[0, 0])
                posterior2 = 1 - posterior1
                probs[i, :] = np.array([posterior1, posterior2])

        else: # <- K-class case
            for i in range(N):
                a = self._W_est[1:, :].T @ x[i, :] + self._W_est[0, :]
                posteriors = softmax(a)
                probs[i, :] = posteriors

        return probs

class LogisticRegression(Classificator):
    def __init__(self, X, t, epochs = 10000, solver = 'ls', verbose = False) -> None:
        self._Xtrain = makeFeatureMatrix(X)
        self._Ttrain = t
        self._TtrainK = self.__makeConvenientTargetVector(t)
        self._epochs = epochs
        self._solver = solver
        self._verbose = verbose
        self._N = self._Xtrain.shape[0]
        self._K = np.unique(self._Ttrain).size
        self._W_est = np.zeros((NDim + 1, self._K), dtype = np.float32)
        self._Sn_est = np.zeros((NDim + 1, NDim + 1), dtype = np.float32)

    def fit(self, learning_rate = 0.1):
        if self._solver == 'ls':
            return self.__fitLeastSquares(learning_rate)
        elif self._solver == 'irls':
            return self.__fitIterativeReweightedLeastSquares(learning_rate)

    def evaluate(self):
        errors = 0
        Pred = self.predict(self._Xtrain)
        for i in range(self._N):
            k = self._Ttrain[i]
            if Pred[i] != k:
                errors += 1
        return 100 - (errors / self._N) * 100

    def predict(self, x):
        Y = None
        if self._K == 2:
            A = lineEq(x, self._W_est)
            Y = sigmoid(A)
            Y[np.where(Y < 0.5)] = 0
            Y[np.where(Y >= 0.5)] = 1
        else:
            A = lineEq(x, self._W_est)
            Y = softmax(A)
            Y = np.argmax(Y, axis = 1)
        return Y

    def predict_proba(self, x):
        Y = None
        if self._K == 2:
            A = lineEq(x, self._W_est)
            Y = sigmoid(A)
        else:
            A = lineEq(x, self._W_est)
            Y = softmax(A)
        return Y

    def predict_poster(self, x):
        if x.ndim == 1 and x.size == 2:
            x = np.insert(x, 0, 1)
        elif x.ndim == 2 and x.shape[1] == 2:
            x = makeFeatureMatrix(x)

        mu_a = self._W_est.T @ x
        stdev2_a = x.T @ self._Sn_est @ x
        kappa = (1 + (np.pi * stdev2_a) / 8)**(-1 / 2)
        return sigmoid(kappa * mu_a)

    def __fitLeastSquares(self, learning_rate = 0.1, lmbd = 0.001):
        verboseInterval = self._epochs // 20
        if verboseInterval == 0:
            verboseInterval = 1
        
        if self._K == 2:
            W_est = np.zeros(NDim + 1, dtype = np.float32)
            for e in range(self._epochs):
                a = lineEq(self._Xtrain, W_est)
                y = sigmoid(a)[:, 0]
                grad_E = self._Xtrain.T @ (y - self._Ttrain) + lmbd * W_est
                grad_E /= self._N
                W_est -= learning_rate * grad_E

                # Bayesian treatment
                S0_inv = np.diag( np.full(NDim + 1, lmbd) )
                self._Sn_est = np.zeros((NDim + 1, NDim + 1), dtype = np.float32)
                for i in range(self._N):
                    self._Sn_est += y[i] * (1 - y[i]) * np.outer(self._Xtrain[i, :], self._Xtrain[i, :])
                self._Sn_est += S0_inv
                self._Sn_est = LA.inv(self._Sn_est)

                if self._verbose and e % verboseInterval == 0:
                    loss = (-self._Ttrain * np.log(y) - (1 - self._Ttrain) * np.log(1 - y)).mean()
                    print(">>> epoch {0}".format(e), "loss: {0}".format(loss), sep = ', ')

        else:
            W_est = np.zeros_like(self._W_est)
            for e in range(self._epochs):
                A = lineEq(self._Xtrain, W_est)
                Y = softmax(A)
                for k in range(self._K):
                    grad_E = self._Xtrain.T @ (Y[:, k] - self._TtrainK[:, k]) + lmbd * W_est
                    grad_E /= self._N
                    W_est[:, k] -= learning_rate * grad_E

                if self._verbose and e % verboseInterval == 0:
                    loss = 0.0
                    for n in range(self._N):
                        for k in range(self._K):
                            loss += self._TtrainK[n, k] * np.log(Y[n, k])
                    loss = -loss / self._N
                    print(">>> epoch {0}".format(e), "loss: {0}".format(loss), sep = ', ')

        self._W_est = W_est
        return self._W_est

    def __fitIterativeReweightedLeastSquares(self, learning_rate = 0.1):
        if self._K == 2:
            verboseInterval = self._epochs // 20
            if verboseInterval == 0:
                verboseInterval = 1
            W_est = np.zeros(NDim + 1, dtype = np.float32)
            for e in range(self._epochs):
                Phi = self._Xtrain
                t = self._Ttrain
                a = lineEq(Phi, W_est)
                y = sigmoid(a)[:, 0]
                R = y * (1 - y)
                R[np.where(R == 0.0)] = 0.000001
                R_inv = 1 / R
                R = np.diag(R)
                R_inv = np.diag(R_inv)
                z = Phi @ W_est - R_inv @ (y - t)
                W_est = LA.inv( Phi.T @ R @ Phi ) @ Phi.T @ R @ z

                if self._verbose and e % verboseInterval == 0:
                    loss = (-self._Ttrain * np.log(y) - (1 - self._Ttrain) * np.log(1 - y)).mean()
                    print(">>> epoch {0}".format(e), "loss: {0}".format(loss), sep = ', ')

        self._W_est = W_est
        return self._W_est

    def __makeConvenientTargetVector(self, t):
        T = np.zeros((t.size, t.max() + 1))
        T[np.arange(t.size), t] = 1
        return T
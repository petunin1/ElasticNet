# https://github.com/petunin1/ElasticNet

import numpy as np
import scipy


def planeRot(x):
    '''
    Orthogonal matrix G for a 2-element vector x such that G x = y where the second element of y is zero.
    :param x: 2-element vector
    :return: matrix G
    '''
    l = np.sqrt(x[0] ** 2 + x[1] ** 2)
    if not l:
        return np.eye(2, dtype = x.dtype), np.full(0.0, dtype = x.dtype)
    g0 = x[0] / l
    g1 = x[1] / l
    return np.array([[g0, g1], [-g1, g0]], dtype = x.dtype), np.array([l, 0.0], dtype = x.dtype)


class Cholesky:
    '''
    Cholesky (upper diagonal-triangular) decomposition C of an initial matrix X, supporting addition and removal of vectors, such that C.T C = X.T X.
    '''

    def __init__(self, dtype, reserve = 0):
        '''
        :param dtype: dtype of the decomposition matrix
        :param reserve: pre-allocated dimension of the decomposition matrix
        '''
        self.R = np.empty([reserve, reserve], dtype = dtype) # upper triangular
        self.points = 0

    def insertProducts(self, XTx, xTx):
        '''
        Addition of a vector x to the existing set X.
        :param XTx: X.T.x
        :param xTx: x.T.x
        '''
        if not self.points:
            if not self.R.shape[0]:
                self.R = np.array([[np.sqrt(xTx)]], dtype = xTx.dtype)
            else:
                self.R[0, 0] = np.sqrt(xTx)
            self.points = 1
            return
        if self.R.shape[0] == self.points:
            R_new = np.empty([self.R.shape[0] + 1, self.R.shape[0] + 1], dtype = self.R.dtype)
            R_new[:self.points, :self.points] = self.R[:self.points, :self.points]
            self.R = R_new
        self.R[:self.points, self.points] = scipy.linalg.solve_triangular(self.get().T, XTx, lower = True, overwrite_b = True)
        self.R[self.points, :self.points] = 0.0
        self.R[self.points, self.points] = np.sqrt(max(0.0, xTx - np.sum(self.R[:self.points, self.points] ** 2)))
        self.points += 1

    def remove(self, j):
        '''
        Removal of a vector from the existing set.
        :param j: index of the vector to be removed in the remaining set
        '''
        self.R[:self.points, j:self.points - 1] = self.R[:self.points, j + 1:self.points]
        self.points -= 1
        for k in range(j, self.points):
            G, y = planeRot(self.R[k:k + 2, k])
            self.R[k:k + 2, k] = y
            if k < self.points - 1:
                self.R[k:k + 2, k + 1:self.points] = G @ self.R[k:k + 2, k + 1:self.points]

    def get(self):
        '''
        :return: Cholesky matrix
        '''
        return self.R[:self.points, :self.points]

    '''
    x = np.random.rand(int(10e4), 3)
    ch = Cholesky(np.float32)
    ch.insertProducts(None, np.sum(x[:, 0] ** 2))
    ch.insertProducts(x[:, [0]].T @ x[:, 2], np.sum(x[:, 1] ** 2))
    ch.insertProducts(x[:, [0, 1]].T @ x[:, 2], np.sum(x[:, 2] ** 2))
    ch.remove(1)
    ch.insertProducts(x[:, [0, 2]].T @ x[:, 1], np.sum(x[:, 1] ** 2))
    print(x.T @ x) # original x.T x matrix
    print(ch.get().T @ ch.get()) # the same for Cholesky decomposition instead of x, rearranged by order of inserting the variables
    '''


def elasticNet(x = None, y = None, xtx = None, xty = None, overwriteMatrices = False, l1 = 0.0, l2 = 0.0, itMax = np.iinfo(np.int64).max, varMax = np.iinfo(np.int64).max, cholesky = False):
    '''
    Least angle implementation of the ElasticNet.
    Finding an optimal beta for a single or multiple values of l1 and a single value of l2.
    One should specify either {x, y} or {xtx, xty}.
    Pre-calculated xtxTotal, xtyTotal, nTotal can be specified for speed.
    :param x: matrix of independent vectors
    :param y: dependent vector or matrix with last dimension of size 1
    :param xtx: x.T.x matrix product divided by the number of observations
    :param xty: x.T.y matrix-vector product divided by the number of observations
    :param overwriteMatrices: whether xtx and xty can be overwritten
    :param l1: L1 regularisation parameter or a list of parameters
    :param l2: L2 regularisation parameter
    :param itMax: maximum number of iterations, including when a variable is removed
    :param varMax: maximum number of dependent variables to be included
    :param cholesky: whether to use Cholesky decomposition for xtx
    :return: vector of betas if l1 is a scalar, matrix of betas with columns corresponding to l1 in descending order otherwise
    '''

    '''
    L = ||y - x * b||_2 ^ 2 / n + l1 * ||b||_1 + l2 * ||b||_2 ^ 2
    X / sqrt(n) -> [X / sqrt(n) ; sqrt(l2) Id]
    L -> ||y - x * b||_2 ^ 2 + l1 * ||b||_1
    '''

    if xtx is None:
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 2:
            y = y[:, 0]
        xty = x.T @ y / len(x)
        xtx = x.T @ x / len(x)
    elif not overwriteMatrices:
        xtx = xtx.copy()
        xty = xty.copy()

    p = len(xtx)
    xtx.flat[::p + 1] += l2
    returnMatrix = True
    if type(l1) == np.ndarray:
        l1List = l1.astype(xtx.dtype)
    elif type(l1) == list:
        l1List = np.array(l1, dtype = xtx.dtype)
    else:
        l1List = np.array([l1], dtype = xtx.dtype)
        returnMatrix = False
    if len(l1List) == 1 and not l1List[0]:
        try:
            beta = scipy.linalg.solve(xtx, xty)
            return beta[:, np.newaxis] if returnMatrix else beta
        except:
            return np.full([p, 1] if returnMatrix else p, 0.0, dtype = xtx.dtype)
    l1List /= 2
    l1List = np.sort(l1List, kind = 'mergesort')[::-1]
    l1 = l1List[0]
    l1Ind = 0
    onePlus = np.float32(1.0) + np.finfo(xtx.dtype).tiny

    if cholesky:
        '''
        \hat X^{T} \hat X = \hat R^{T} \hat R    [R is upper-triangular]
        '''
        ch = Cholesky(xtx.dtype, p)
    betaList = np.full((p, len(l1List)), 0.0, dtype = xtx.dtype)
    beta = np.full(p, 0.0, dtype = xtx.dtype)
    I = np.full(p, True, dtype = np.bool)
    A = np.empty(0, dtype = np.int64)
    indDrop = 0

    dropVariable = False
    it = 0
    var = 0

    if p < varMax:
        varMax = p
    while var < varMax and it != itMax:
        j = np.where(I)[0][np.argmax(np.fabs(xty[I]))]
        # print('+', j)
        C = np.fabs(xty[j])

        if not dropVariable:
            if cholesky:
                ch.insertProducts(xtx[A, j], xtx[j, j])
            A = np.concatenate([A, np.array([j], dtype = np.int64)], axis = 0)
            I[j] = False
            var += 1

        s = np.sign(xty[A])
        '''
        \vec d = \hat X \vec g    [\vec d - new direction, \hat X - active inputs, \vec g - weights of inputs in the new direction]
        \hat X^{T} \vec d = \vec s    [\vec s - sign of the vector product of \hat X and the unfitted remainder \vec r]
        \hat X^{T} \hat X \vec g = \vec s
        '''
        g = scipy.linalg.solve_triangular(ch.get(), scipy.linalg.solve_triangular(ch.get().T, s, lower = True), lower = False, overwrite_b = True) if cholesky else \
            scipy.linalg.solve(xtx[A, :][:, A], s)
        a = xtx[I, :][:, A] @ g # X(inactive) & d vector product with=without regularisation (the same as "x[:, I].T @ d")

        '''
        (\pm \vec X_{A} \pm_{2} \vec X_{I}) (\vec y - \gamma \vec d) = 0    [ignore inactive inputs to get the gamma unconstrained by inactive inputs]
        \gamma (1 \pm a_{I}) = C \pm c_{I}
        '''

        gamma = C  # unconstrained

        gammaAdd = np.concatenate([(C - xty[I]) / (onePlus - a), (C + xty[I]) / (onePlus + a)], axis = 0)  # gamma where another input should be added
        gammaAdd = gammaAdd[gammaAdd > 0]
        if len(gammaAdd):
            gammaAdd = np.min(gammaAdd)
            if gammaAdd < gamma:
                gamma = gammaAdd

        dropVariable = False
        gammaRemove = -beta[A] / g  # gamma where an input should be removed
        gammaRemovePositive = gammaRemove[gammaRemove > 0]
        if len(gammaRemovePositive):
            gammaRemoveMin = np.min(gammaRemovePositive)
            if gammaRemoveMin < gamma:
                gamma = gammaRemoveMin
                indDrop = np.where(gammaRemove == gamma)[0][0]
                dropVariable = True

        g *= gamma
        betaActiveNew = beta[A] + g

        while True:
            # "np.sum((y - x @ beta)) * d)" = "xty @ beta)"
            # "np.sum(d ** 2)" = "g @ xtx[A, :][:, A] @ g"
            fraction = (xty[A] @ g - l1 * np.sum(np.fabs(betaActiveNew - beta[A]))) / (g @ xtx[A, :][:, A] @ g)
            if fraction > 1:
                break
            betaList[:, l1Ind] = beta
            if fraction > 0:
                betaList[A, l1Ind] += fraction * g
            l1Ind += 1
            if l1Ind == len(l1List):
                return betaList if returnMatrix else betaList[:, 0]
            l1 = l1List[l1Ind]

        beta[A] = betaActiveNew
        xty -= xtx[:, A] @ g
        it += 1

        if dropVariable:
            if cholesky:
                ch.remove(indDrop)
            ind = A[indDrop]
            I[ind] = True
            # print('-', A[indDrop])
            beta[ind] = 0
            A = np.concatenate((A[:indDrop], A[indDrop + 1:]), axis = 0)
            var -= 1

    for col in range(l1Ind, len(l1List)):
        betaList[:, col] = beta

    return betaList if returnMatrix else betaList[:, 0]


def elasticNetCoordinateDescent(x = None, y = None, xtx = None, xty = None, overwriteMatrices = False, l1 = None, l2 = None, tol = 1e-4, itMax = 1000):
    '''
    Coordinate descent implementation of the ElasticNet.
    One should specify either {x, y} or {xtx, xty}.
    :param x: matrix of independent vectors
    :param y: dependent vector or matrix with last dimension of size 1
    :param xtx: x.T.x matrix product divided by the number of observations
    :param xty: x.T.y matrix-vector product divided by the number of observations
    :param overwriteMatrices: whether xtx and xty can be overwritten
    :param l1: L1 regularisation parameter
    :param l2: L2 regularisation parameter
    :param tol: tolerance of beta components relative to the largest one, used as a stopping parameter
    :param itMax: maximum number of iterations of running through all beta components
    :return: vector of betas
    '''

    '''
    L = ||y - x * b||_2 ^ 2 / n + l1 * ||b||_1 + l2 * ||b||_2 ^ 2
    L1 = ||r - x * b1||_2 ^ 2 / n + l1 * |b1| + l2 * b1 ^ 2
    b1 = max(0, r * x / n - l1 / 2) * sign(r * x) / (x ^ 2 / n + l2)
    '''

    if xtx is None:
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 2:
            y = y[:, 0]
        xty = x.T @ y / len(x)
        xtx = x.T @ x / len(x)
    elif not overwriteMatrices:
        xtx = xtx.copy()
        xty = xty.copy()

    #@nb.njit(nogil = True, fastmath = True)
    def elasticNetCoordinateDescent1(xtx, xty, l1, l2, tol, itMax):
        p = len(xtx)
        l1 /= 2
        normSquareInverse = (1.0 / (xtx.flat[::p + 1] + l2[0])).astype(xtx.dtype)
        beta = np.full(p, 0.0, dtype = xtx.dtype)
        for it in range(itMax):
            shiftMax = 0.0
            for i in range(p):
                b1Old = beta[i]
                beta[i] = 0.0
                prod = xty[i] - xtx[i] @ beta
                b1 = np.fabs(prod) - l1[0]
                if b1 <= 0.0:
                    continue
                b1 *= np.sign(prod) * normSquareInverse[i]
                shiftMax = max(shiftMax, np.fabs(b1 - b1Old))
                beta[i] = b1
            betaAbsMax = np.max(np.fabs(beta))
            if not betaAbsMax or shiftMax / betaAbsMax < tol[0]:
                break
        return beta

    return elasticNetCoordinateDescent1(xtx = xtx, xty = xty, l1 = np.array([l1]).astype(xtx.dtype), l2 = np.array([l2]).astype(xtx.dtype), tol = np.array([tol]).astype(xtx.dtype), itMax = itMax)


def elasticNetCV(x = None, y = None, batches = None, xtx = None, xty = None, n = None, l1 = np.logspace(-15.0, 5.0, num = 100, base = 2.0), l2 = np.logspace(-15.0, 5.0, num = 30, base = 2.0), itMax = np.iinfo(np.int64).max, varMax = np.iinfo(np.int64).max, cholesky = False, returnBeta = True):
    '''
    Cross-validated version of the elasticNet function.
    Finding an optimal beta, l1, l2, using the elasticNet function.
    One should either specify {x, y, batches} or {xtx, xty, n}.
    :param x: matrix of independent vectors
    :param y: dependent vector or matrix with last dimension of size 1
    :param batches: number of batches for cross-validation (should be >=2)
    :param xtx: 3-dimensional tensor of x.T.x matrix products where the last dimension corresponds to batches
    :param xty: 2-dimensional tensor of x.T.y matrix-vector products where the last dimension corresponds to batches
    :param n: vector of the number of data points in each batch
    :param nTotal: total number of data points
    :param l1: L1 regularisation parameter or a list of parameters
    :param l2: L2 regularisation parameter or a list of parameters
    :param itMax: maximum number of iterations of adding and removing variables along the L1 path
    :param varMax: maximum number of variables in the resulting set
    :param cholesky: whether to use Cholesky decomposition for finding betas
    :param returnBeta: whether to return optimal beta
    :return: optimal beta for the combined set of data; optimal L1; optimal L2
    '''

    if xtx is None:
        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 2:
            y = y[:, 0]
        nTotal, p = x.shape
        batchLen = np.int64(np.floor(nTotal / batches))
        xtx = np.empty([p, p, batches], dtype = x.dtype)
        xty = np.empty([p, batches], dtype = xtx.dtype)
        n = np.full(batches, batchLen, dtype = np.int64)
        begin = 0
        end = batchLen
        for b in range(batches):
            if b == batches - 1:
                end = nTotal
                n[b] = end - begin
            xb = x[begin:end]
            xtx[:, :, b] = xb.T @ xb
            yb = y[begin:end]
            xty[:, b] = xb.T @ yb
            begin += batchLen
            end += batchLen
    else:
        nTotal = np.sum(n)
    xtxTotal = np.sum(xtx, axis = 2)
    xtyTotal = np.sum(xty, axis = 1)

    l1List = np.sort(np.array(l1, dtype = xtx.dtype) if type(l1) == list else l1 if type(l1) == np.ndarray else np.array([l1], dtype = xtx.dtype), kind = 'mergesort')[::-1]
    l2List = np.array(l2, dtype = xtx.dtype) if type(l2) == list else l2 if type(l2) == np.ndarray else np.array([l2], dtype = xtx.dtype)
    l1Len = len(l1List)
    l2Len = len(l2List)
    cost = np.full([l1Len, l2Len], 0.0, dtype = xtx.dtype)

    for b in range(len(n)):
        for l2Ind in range(l2Len):
            beta = elasticNet(x = None, y = None, xtx = (xtxTotal - xtx[:, :, b]) / (nTotal - n[b]), xty = (xtyTotal - xty[:, b]) / (nTotal - n[b]), overwriteMatrices = True, l1 = l1List, l2 = l2List[l2Ind], cholesky = cholesky)
            cost[:, l2Ind] += np.sum(np.tensordot(xtx[:, :, b], beta, axes = [1, 0]) * beta, axis = 0) - 2 * np.tensordot(xty[:, b], beta, axes = [0, 0])

    l1Ind, l2Ind = np.unravel_index(np.argmin(cost), cost.shape)
    l1 = l1List[l1Ind]
    l2 = l2List[l2Ind]
    return elasticNet(x = None, y = None, xtx = xtxTotal / nTotal, xty = xtyTotal / nTotal, overwriteMatrices = True, l1 = l1, l2 = l2) if returnBeta else None, l1, l2


def generateRandomRegression(n, p):
    '''
    Generation of random variables for a regression.
    :param n: number of data points
    :param p: number of dependent variables
    :return: matrix of independent variables, vector of dependent variables
    '''
    n = int(n)
    p = int(p)
    x = 3.0 * (np.random.rand(n, p) - 0.5)
    y = x.dot(np.random.rand(p) - 0.5) + np.random.rand(n)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

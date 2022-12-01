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
        :return:
        '''
        if not self.points:
            if not self.R.shape[0]:
                self.R = np.array([[np.sqrt(xTx)]], dtype = x.dtype)
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
                self.R[k:k + 2, k + 1:self.points] = np.dot(G, self.R[k:k + 2, k + 1:self.points])

    def get(self):
        return self.R[:self.points, :self.points]

    '''
    x = np.random.rand(10e4, 3)
    ch = Cholesky(np.float32)
    ch.insertProducts(None, np.sum(x[:, 0] ** 2))
    ch.insertProducts(np.dot(x[:, [0]].T, x[:, 2]), np.sum(x[:, 1] ** 2))
    ch.insertProducts(np.dot(x[:, [0, 1]].T, x[:, 2]), np.sum(x[:, 2] ** 2))
    ch.remove(1)
    ch.insert(x[:, [0, 2]], x[:, 1])
    print(np.dot(x.T, x)) # original x.T x matrix 
    print(np.dot(ch.get().T, ch.get())) # the same for Cholesky decomposition instead of x
    '''


def elasticNet(x = None, y = None, n = None, xtx = None, xty = None, l1 = 0.0, l2 = 0.0, itMax = np.iinfo(np.int64).max, varMax = np.iinfo(np.int64).max, cholesky = False):
    '''
    Least angle implementation of the ElasticNet.
    Finding an optimal beta for a single or multiple values of l1 and a single value of l2.
    One should either specify {x, y} or {xtx, xty, n}.
    Pre-calculated xtxTotal, xtyTotal, nTotal can be specified for speed.
    :param x: matrix of independent vectors
    :param y: dependent vector or matrix with last dimension of size 1
    :param n: number of data points
    :param xtx: x.T.x matrix product
    :param xty: x.T.y matrix-vector product
    :param l1: L1 regularisation parameter or a list of parameters
    :param l2: L2 regularisation parameter
    :param itMax: maximum number of iterations, including when a variable is removed
    :param varMax: maximum number of dependent variables to be included
    :param cholesky: whether to use Cholesky decomposition for xtx
    :return:
    '''
    # l1 can be a list of non-negative values (sorted from highest); xtx and xty will be overwritten
    '''
    l1 -> l1 * n
    l2 -> l2 * n
    L = ||y - x * b||_2 ^ 2 + l1 * ||b||_1 + l2 * ||b||_2 ^ 2
    X -> [X ; sqrt(l2) Id]
    L -> ||y - x * b||_2 ^ 2 + l1 * ||b||_1
    '''

    if x is not None:
        n = len(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]
    if y is not None and y.ndim == 2:
        y = y[:, 0]

    if xtx is None:
        xtx = np.dot(x.T, x)
    p = len(xtx)
    l2 *= n
    xtx.flat[::p + 1] += l2
    returnMatrix = True
    if xty is None:
        xty = np.dot(x.T, y)
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
    l1List *= n / 2
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
    if not l2 and n < varMax:
        varMax = n
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
        a = np.dot(xtx[I, :][:, A], g) # X(inactive) & d vector product with=without regularisation (the same as "np.dot(x[:, I].T, d)")

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
            # "np.sum((y - np.dot(x, beta)) * d)" = "np.dot(xty, beta)"
            # "np.sum(d ** 2)" = "np.dot(g, np.dot(xtx[A, :][:, A], g))"
            fraction = (np.dot(xty[A], g) - l1 * np.sum(np.fabs(betaActiveNew - beta[A]))) / np.dot(g, np.dot(xtx[A, :][:, A], g))
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
        xty -= np.dot(xtx[:, A], g)
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


def elasticNetCoordinateDescent(x, y, l1, l2, tol = 1e-4, itMax = 1000):
    '''
    :param x: matrix of independent vectors
    :param y: dependent vector or matrix with last dimension of size 1
    :param l1: L1 regularisation parameter
    :param l2: L2 regularisation parameter
    :param tol: tolerance of beta components relative to the largest one, used as a stopping parameter
    :param itMax: maximum number of iterations of running through all beta components
    '''

    '''
    l1 -> l1 * n
    l2 -> l2 * n
    L = ||y - x * b||_2 ^ 2 + l1 * ||b||_1 + l2 * ||b||_2 ^ 2
    L1 = ||r - x * b1||_2 ^ 2 + l1 * |b1| + l2 * b1 ^ 2
    b1 = max(0, r * x - l1 / 2) * sign(r * x) / (x ^ 2 + l2)
    '''

    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 2:
        y = y[:, 0]

    #@nb.njit(nogil = True, fastmath = True)
    def elasticNetCoordinateDescent1(x, y, l1, l2, tol, itMax):
        n, p = x.shape

        xty = np.dot(x.T, y)
        xtx = np.dot(x.T, x)

        l1 *= n / 2
        l2 *= n

        normSquareInverse = (1.0 / (np.sum(x ** 2, axis = 0) + l2[0])).astype(x.dtype)

        r = y.copy()
        beta = np.full(p, 0.0, dtype = y.dtype)

        for it in range(itMax):
            shiftMax = 0.0
            for i in range(p):
                if beta[i]:
                    r += x[:, i] * beta[i]
                b1Old = beta[i]
                beta[i] = 0.0
                prod = xty[i] - np.dot(xtx[i], beta) #np.dot(r, x1) #tf.tensordot(r, x1, [0, 0]).numpy()
                b1 = np.fabs(prod) - l1[0]
                if b1 <= 0.0:
                    continue
                b1 *= np.sign(prod) * normSquareInverse[i]
                shiftMax = max(shiftMax, np.fabs(b1 - b1Old))
                beta[i] = b1
                r -= x[:, i] * b1
            betaAbsMax = np.max(np.fabs(beta))
            if not betaAbsMax or shiftMax / betaAbsMax < tol[0]:
                break

        return beta

    beta = elasticNetCoordinateDescent1(x, y.astype(x.dtype), np.array([l1]).astype(x.dtype), np.array([l2]).astype(x.dtype), np.array([tol]).astype(x.dtype), itMax)

    return beta


def elasticNetCV(x = None, y = None, batches = None, xtx = None, xty = None, n = None, xtxTotal = None, xtyTotal = None, nTotal = None, l1 = np.logspace(-15.0, 5.0, num = 100, base = 2.0), l2 = np.logspace(-15.0, 5.0, num = 30, base = 2.0), itMax = np.iinfo(np.int64).max, varMax = np.iinfo(np.int64).max, cholesky = False, returnBeta = True):
    '''
    Cross-validated version of the elasticNet function.
    Finding an optimal beta, l1, l2, using the elasticNet function.
    One should either specify {x, y, batches} or {xtx, xty, n}.
    Pre-calculated xtxTotal, xtyTotal, nTotal can be specified for speed.
    :param x: matrix of independent vectors
    :param y: dependent vector or matrix with last dimension of size 1
    :param batches: number of batches for cross-validation (should be >=2)
    :param xtx: 3-dimensional tensor of x.T.x matrix products where the last dimension corresponds to batches
    :param xty: 2-dimensional tensor of x.T.y matrix-vector products where the last dimension corresponds to batches
    :param n: vector of the number of data points in each batch
    :param xtxTotal: sum of xtx across the batches
    :param xtyTotal: sum of xty across the batches
    :param nTotal: total number of data points
    :param l1: L1 regularisation parameter or a list of parameters
    :param l2: L2 regularisation parameter or a list of parameters
    :param itMax: maximum number of iterations of adding and removing variables along the L1 path
    :param varMax: maximum number of variables in the resulting set
    :param cholesky: whether to use Cholesky decomposition for finding betas
    :param returnBeta: whether to return optimal beta
    :return: optimal beta for the combined set of data; optimal L1; optimal L2
    '''
    if x is not None and y is not None and batches is not None:
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
            xtx[:, :, b] = np.dot(xb.T, xb)
            yb = y[begin:end]
            xty[:, b] = np.dot(xb.T, yb)
            begin += batchLen
            end += batchLen
    if xtxTotal is None:
        xtxTotal = np.sum(xtx, axis = 2)
    if xtyTotal is None:
        xtyTotal = np.sum(xty, axis = 1)
    if nTotal is None:
        nTotal = np.sum(n)

    l1List = np.sort(np.array(l1, dtype = xtx.dtype) if type(l1) == list else l1 if type(l1) == np.ndarray else np.array([l1], dtype = xtx.dtype), kind = 'mergesort')[::-1]
    l2List = np.array(l2, dtype = xtx.dtype) if type(l2) == list else l2 if type(l2) == np.ndarray else np.array([l2], dtype = xtx.dtype)
    l1Len = len(l1List)
    l2Len = len(l2List)
    cost = np.full([l1Len, l2Len], 0.0, dtype = xtx.dtype)

    for b in range(len(n)):
        xtxBatch = xtxTotal - xtx[:, :, b]
        xtyBatch = xtyTotal - xty[:, b]
        nBatch = nTotal - n[b]
        for l2Ind in range(l2Len):
            beta = elasticNet(x = None, y = None, n = nBatch, xtx = xtxBatch.copy(), xty = xtyBatch.copy(), l1 = l1List, l2 = l2List[l2Ind], cholesky = cholesky)
            cost[:, l2Ind] += np.sum(np.tensordot(xtx[:, :, b], beta, axes = [1, 0]) * beta, axis = 0) - 2 * np.tensordot(xty[:, b], beta, axes = [0, 0])

    l1Ind, l2Ind = np.unravel_index(np.argmin(cost), cost.shape)
    l1 = l1List[l1Ind]
    l2 = l2List[l2Ind]
    #print(cost)
    return elasticNet(x = None, y = None, n = nTotal, xtx = xtxTotal, xty = xtyTotal, l1 = l1, l2 = l2) if returnBeta else None, l1, l2

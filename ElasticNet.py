import numpy as np
import datetime


def rand(*args):
    return np.random.rand(*[int(a) for a in args])


def p2u(p):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (p - epoch).total_seconds() * 1000.0


def now():
    return p2u(datetime.datetime.utcnow())


def cat(*args, **kwargs):
    return np.concatenate([a for a in args], **kwargs)


def dimPush(a):
    return a.reshape(cat(a.shape, [1]))


def dimPop(a):
    return a.reshape(a.shape[:-1])


def planeRot(x):
    '''
    G x = y
    G - orthogonal 2x2 matrix, x & y : 2 vectors, y[2] = 0
    return G, y
    '''
    l = np.sqrt(x[0] ** 2 + x[1] ** 2)
    if not l:
        return np.eye(2, dtype = x.dtype), np.full(0.0, dtype = x.dtype)
    g0 = x[0] / l
    g1 = x[1] / l
    return np.array([[g0, g1], [-g1, g0]], dtype = x.dtype), np.array([l, 0.0], dtype = x.dtype)


class Cholesky:
    def __init__(self, dtype, reserve = 0):
        self.R = np.empty([reserve, reserve], dtype = dtype) # upper triangular
        self.points = 0

    def insert(self, X, x, l2 = 0.0):
        xTx = np.sum(x ** 2) + l2
        if not self.points:
            if not self.R.shape[0]:
                self.R = np.array([[np.sqrt(xTx)]], dtype = x.dtype)
            else:
                self.R[0, 0] = np.sqrt(xTx)
            self.points = 1
            return
        from scipy import linalg
        if self.R.shape[0] == self.points:
            R_new = np.empty([self.R.shape[0] + 1, self.R.shape[0] + 1], dtype = self.R.dtype)
            R_new[:self.points, :self.points] = self.R[:self.points, :self.points]
            self.R = R_new
        self.R[:self.points, self.points] = dimPop(linalg.solve_triangular(self.get().T, np.dot(X.T, dimPush(x)), lower = True, overwrite_b = True))
        self.R[self.points, :self.points] = 0.0
        self.R[self.points, self.points] = np.sqrt(max(0.0, xTx - np.sum(self.R[:self.points, self.points] ** 2)))
        self.points += 1

    def insertProducts(self, XTx, xTx):
        if not self.points:
            if not self.R.shape[0]:
                self.R = np.array([[np.sqrt(xTx)]], dtype = x.dtype)
            else:
                self.R[0, 0] = np.sqrt(xTx)
            self.points = 1
            return
        from scipy import linalg
        if self.R.shape[0] == self.points:
            R_new = np.empty([self.R.shape[0] + 1, self.R.shape[0] + 1], dtype = self.R.dtype)
            R_new[:self.points, :self.points] = self.R[:self.points, :self.points]
            self.R = R_new
        self.R[:self.points, self.points] = linalg.solve_triangular(self.get().T, XTx, lower = True, overwrite_b = True)
        self.R[self.points, :self.points] = 0.0
        self.R[self.points, self.points] = np.sqrt(max(0.0, xTx - np.sum(self.R[:self.points, self.points] ** 2)))
        self.points += 1

    def remove(self, j):
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
    x = rand(10e4, 3)
    ch = Cholesky(np.float32)
    ch.insert(None, x[:, 0])
    ch.insert(x[:, [0]], x[:, 1])
    ch.insertProducts(np.dot(x[:, [0, 1]].T, x[:, 2]), np.sum(x[:, 2] ** 2))
    ch.remove(1)
    ch.insert(x[:, [0, 2]], x[:, 1])
    print(np.dot(ch.get().T, ch.get()))
    print(np.dot(x.T, x))
    '''


def getBeta(x = None, y = None, n = None, xtx = None, xty = None, l1 = 0.0, l2 = 0.0, itMax = np.iinfo(np.int64).max, varMax = np.iinfo(np.int64).max, cholesky = False):
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
            beta = np.linalg.solve(xtx, xty)
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

    from scipy import linalg

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
        g = linalg.solve_triangular(ch.get(), linalg.solve_triangular(ch.get().T, s, lower = True), lower = False, overwrite_b = True) if cholesky else \
            np.linalg.solve(xtx[A, :][:, A], s)
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


def getBetaCoordinateDescent(x, y, l1, l2, tol = 1e-4, itMax = 1000):
    '''
    l1 -> l1 * n
    l2 -> l2 * n
    L = ||y - x * b||_2 ^ 2 + l1 * ||b||_1 + l2 * ||b||_2 ^ 2
    L1 = ||r - x * b1||_2 ^ 2 + l1 * |b1| + l2 * b1 ^ 2
    b1 = max(0, r * x - l1 / 2) * sign(r * x) / (x ^ 2 + l2)
    '''

    if x.ndim == 1:
        x = dimPush(x)
    if y.ndim == 2:
        y = dimPop(y)

    #@nb.njit(nogil = True, fastmath = True)
    def getBetaCoordinateDescent1(x, y, l1, l2, tol, itMax):
        n, p = x.shape

        xy = np.dot(x.T, y)
        xx = np.dot(x.T, x)

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
                prod = xy[i] - np.dot(xx[i], beta) #np.dot(r, x1) #tf.tensordot(r, x1, [0, 0]).numpy()
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

    beta = getBetaCoordinateDescent1(x, y.astype(x.dtype), np.array([l1]).astype(x.dtype), np.array([l2]).astype(x.dtype), np.array([tol]).astype(x.dtype), itMax)

    return beta


def getBetaCV(x = None, y = None, n = None, xtx = None, xty = None, xtxTotal = None, xtyTotal = None, l1 = np.logspace(-15.0, 5.0, num = 100, base = 2.0), l2 = np.logspace(-15.0, 5.0, num = 30, base = 2.0), batches = 3, itMax = np.iinfo(np.int64).max, varMax = np.iinfo(np.int64).max, cholesky = False):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 2:
        y = y[:, 0]
    if x is not None:
        nTotal, p = x.shape

    batchLen = np.int64(np.floor(nTotal / batches))
    if xtx is None:
        nList = np.full(batches, batchLen, dtype = np.int64)
        xtx = np.empty([p, p, batches], dtype = x.dtype)
        begin = 0
        end = batchLen
        for b in range(batches):
            if b == batches - 1:
                end = nTotal
                nList[b] = end - begin
            xb = x[begin:end]
            xtx[:, :, b] = np.dot(xb.T, xb)
            begin += batchLen
            end += batchLen
    if xty is None:
        xty = np.empty([p, batches], dtype = xtx.dtype)
        begin = 0
        end = batchLen
        for b in range(batches):
            if b == batches - 1:
                end = nTotal
            xb = x[begin:end]
            yb = y[begin:end]
            xty[:, b] = np.dot(xb.T, yb)
            begin += batchLen
            end += batchLen

    if xtxTotal is None:
        xtxTotal = np.sum(xtx, axis = 2)
    if xtyTotal is None:
        xtyTotal = np.sum(xty, axis = 1)

    l1List = np.sort(np.array(l1, dtype = xtx.dtype) if type(l1) == list else l1 if type(l1) == np.ndarray else np.array([l1], dtype = xtx.dtype), kind = 'mergesort')[::-1]
    l2List = np.array(l2, dtype = xtx.dtype) if type(l2) == list else l2 if type(l2) == np.ndarray else np.array([l2], dtype = xtx.dtype)
    l1Len = len(l1List)
    l2Len = len(l2List)
    cost = np.full([l1Len, l2Len], 0.0, dtype = xtx.dtype)

    for b in range(batches):
        xtxBatch = xtxTotal - xtx[:, :, b]
        xtyBatch = xtyTotal - xty[:, b]
        nBatch = nTotal - nList[b]
        for l2Ind in range(l2Len):
            beta = getBeta(x = None, y = None, n = nBatch, xtx = xtxBatch.copy(), xty = xtyBatch.copy(), l1 = l1List, l2 = l2List[l2Ind], cholesky = cholesky)
            cost[:, l2Ind] += np.sum(np.tensordot(xtx[:, :, b], beta, axes = [1, 0]) * beta, axis = 0) - 2 * np.tensordot(xty[:, b], beta, axes = [0, 0])

    l1Ind, l2Ind = np.unravel_index(np.argmin(cost), cost.shape)
    l1 = l1List[l1Ind]
    l2 = l2List[l2Ind]
    #print(cost)
    return getBeta(x = None, y = None, n = nTotal, xtx = xtxTotal, xty = xtyTotal, l1 = l1, l2 = l2), l1, l2

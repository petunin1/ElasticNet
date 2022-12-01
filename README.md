


```
# generate random variables
def getRandom(n, p):
    n = int(n)
    p = int(p)
    x = 3.0 * (np.random.rand(n, p) - 0.5)
    y = x.dot(np.random.rand(p)) + rand(n)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return x, y

# measuring time
def p2u(p):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (p - epoch).total_seconds() * 1000.0
def now():
    return p2u(datetime.datetime.utcnow())
```

# Basic version:

```
x, y = getRandom(1e7, 100)
l1 = 0.3
l2 = 0.5

t0 = now()
beta = elasticNet(x = x, y = y, l1 = l1, l2 = l2, cholesky = False)
# beta = elasticNet(xtx = x.T.dot(x), xty = x.T.dot(y), n = len(x), l1 = l1, l2 = l2, cholesky = False) # alternative way to get the answer by specifying xtx and xty products
print('ElasticNet.elasticNet latency:    {:.0f}ms'.format(now() - t0))
```

#### Comparison with sklearn:

```
from sklearn.linear_model import ElasticNet as sklearnElasticNet
t0 = now()
en = sklearnElasticNet(alpha = l1 / 2 + l2, l1_ratio = l1 / 2 / (l1 / 2 + l2), fit_intercept = False, max_iter = 1000);
en.fit(x, y)
beta2 = en.coef_
print('sklearnElasticNet latency:    {:.0f}ms'.format(now() - t0))
print(np.concatenate([[beta], [beta2]], axis = 0).T) # compare the betas
```

# Cross-validated version:

```
x, y = getRandom(1e5, 100)

b, l1Optimal, l2Optimal = elasticNetCV(x = x, y = y, batches = 3, l1 = np.logspace(-15.0, 5.0, num = 100, base = 2.0), l2 = np.logspace(-15.0, 5.0, num = 30, base = 2.0), cholesky = False)
```

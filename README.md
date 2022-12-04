




```
# measuring time
import datetime
def p2u(p):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (p - epoch).total_seconds() * 1000.0
def now():
    return p2u(datetime.datetime.utcnow())
```

# Basic version:

```
x, y = generateRandomRegression(1e7, 100)
l1 = 0.5
l2 = 0.3
```

#### Least angles

```
t0 = now()
beta = elasticNet(x = x, y = y, l1 = l1, l2 = l2)
# beta = elasticNet(xtx = x.T.dot(x) / len(x), xty = x.T.dot(y) / len(x), l1 = l1, l2 = l2, cholesky = False) # alternative way to get the answer by specifying xtx and xty products
print('elasticNet latency:    {:.0f}ms'.format(now() - t0))
```

#### Coordinate descent

```
t0 = now()
beta1 = elasticNetCoordinateDescent(x = x, y = y, l1 = l1, l2 = l2)
print('elasticNetCoordinateDescent latency:    {:.0f}ms'.format(now() - t0))
```

#### Comparison with sklearn

```
from sklearn.linear_model import ElasticNet as sklearnElasticNet
t0 = now()
en = sklearnElasticNet(alpha = l1 / 2 + l2, l1_ratio = l1 / 2 / (l1 / 2 + l2), fit_intercept = False, max_iter = 1000);
en.fit(x, y)
beta2 = en.coef_
print('sklearn.ElasticNet latency:    {:.0f}ms'.format(now() - t0))
#print(np.concatenate([[beta], [beta1], [beta2]], axis = 0).T) # compare the betas
```

# Cross-validated version:

```
x, y = generateRandomRegression(1e5, 100)
b, l1Optimal, l2Optimal = elasticNetCV(x = x, y = y, batches = 3, l1 = np.logspace(-15.0, 5.0, num = 100, base = 2.0), l2 = np.logspace(-15.0, 5.0, num = 30, base = 2.0))
```

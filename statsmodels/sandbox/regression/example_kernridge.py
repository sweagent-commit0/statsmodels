import numpy as np
import matplotlib.pyplot as plt
from .kernridgeregress_class import GaussProcess, kernel_euclid
m, k = (50, 4)
upper = 6
scale = 10
xs = np.linspace(1, upper, m)[:, np.newaxis]
xs1 = np.sin(xs)
y1true = np.sum(xs1 + 0.01 * np.sqrt(np.abs(xs1)), 1)[:, np.newaxis]
y1 = y1true + 0.1 * np.random.randn(m, 1)
stride = 3
xstrain = xs1[::stride, :]
ystrain = y1[::stride, :]
xstrain = np.r_[xs1[:m / 2, :], xs1[m / 2 + 10:, :]]
ystrain = np.r_[y1[:m / 2, :], y1[m / 2 + 10:, :]]
index = np.hstack((np.arange(m / 2), np.arange(m / 2 + 10, m)))
gp1 = GaussProcess(xstrain, ystrain, kernel=kernel_euclid, ridgecoeff=5 * 0.0001)
yhatr1 = gp1.predict(xs1)
plt.figure()
plt.plot(y1true, y1, 'bo', y1true, yhatr1, 'r.')
plt.title('euclid kernel: true y versus noisy y and estimated y')
plt.figure()
plt.plot(index, ystrain.ravel(), 'bo-', y1true, 'go-', yhatr1, 'r.-')
plt.title('euclid kernel: true (green), noisy (blue) and estimated (red) ' + 'observations')
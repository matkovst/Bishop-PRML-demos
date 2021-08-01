import numpy as np
from matplotlib import pyplot as plt

data = np.array([
7.05336032e-24, 2.44839765e-16, 6.34758237e-10, 1.53751012e-09,
 8.04391422e-10, 4.45428977e-10, 2.92550319e-10, 2.09794529e-10,
 1.56009885e-10, 1.17863933e-10
])

xp = np.arange(1, data.size + 1)

plt.plot(xp, data)
plt.show()
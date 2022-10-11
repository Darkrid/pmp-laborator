import numpy as np
import itertools
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

omega_1 = stats.expon.rvs(0, 1/4, size=10000)
omega_2 = stats.expon.rvs(0, 1/6, size=10000)
x = stats.binom.rvs(1, 0.4, size=10000)

a = []
for i in range(10000):
    if (x[i] == 1):
        a.append(omega_1[i])
    if (x[i] == 0):
        a.append(omega_2[i])

meann = np.mean(a)
stdd = np.std(a)
print(stdd)
omega = stats.expon.rvs(0, stdd, size=10000)

az.plot_posterior({'Timp medie':omega})
plt.show()
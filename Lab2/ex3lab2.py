import numpy as np
import itertools
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

omega_1 = stats.expon.rvs(0, 1/2, size=10)
omega_2 = stats.expon.rvs(0, 1/3, size=10)
x = stats.binom.rvs(1, 0.5, size=100)
y = stats.binom.rvs(1, 0.3, size=100)

a = []
b = []
c = []
d = []

for i in range(100):
    if (x[i] == 0 and y[i] == 1):
        a.append(1)
        b.append(0)
        c.append(0)
        d.append(0)
    if (x[i] == 0 and y[i] == 0):
        a.append(0)
        b.append(1)
        c.append(0)
        d.append(0)
    if (x[i] == 1 and y[i] == 1):
        a.append(0)
        b.append(0)
        c.append(1)
        d.append(0)
    if (x[i] == 1 and y[i] == 0):    
        a.append(0)
        b.append(0)
        c.append(0)
        d.append(1)

mean_a = np.mean(a)
mean_b = np.mean(b)
mean_c = np.mean(c)
mean_d = np.mean(d)

std_a = np.std(a)
std_b = np.std(b)
std_c = np.std(c)
std_d = np.std(d)

omega_a = stats.expon.rvs(0, std_a, size=100)
omega_b = stats.expon.rvs(0, std_b, size=100)
omega_c = stats.expon.rvs(0, std_c, size=100)
omega_d = stats.expon.rvs(0, std_d, size=100)

az.plot_posterior({'Aruncari Preferabile A':omega_a, 'Aruncari Preferabile B':omega_b, 'Aruncari Preferabile C':omega_c, 'Aruncari Preferabile D':omega_d})
plt.show()
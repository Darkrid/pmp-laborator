from pickletools import read_long1
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

time1 = stats.gamma.rvs(4, 0, 1/3, size=10000)
time2 = stats.gamma.rvs(4, 0, 1/2, size=10000)
time3 = stats.gamma.rvs(5, 0, 1/2, size=10000)
time4 = stats.gamma.rvs(5, 0, 1/3, size=10000)

red1 = stats.gamma.rvs(4, 0, 1/3, size=1)
red2 = stats.gamma.rvs(4, 0, 1/2, size=1)
red3 = stats.gamma.rvs(4, 0, 1/2, size=1)
red4 = stats.gamma.rvs(4, 0, 1/3, size=1)

x = stats.uniform.rvs(0, 1, size=10000)

a = []

for i in range(10000):
    if (x[i] < red1):
        a.append(time1[i])
    if (x[i] < red2):
        a.append(time2[i])
    if (x[i] < red3):
        a.append(time3[i])
    if (x[i] < red4):
        a.append(time4[i])

az.plot_posterior({'Medie': a})
plt.show()
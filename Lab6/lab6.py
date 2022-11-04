import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm

mom_age = []
ppvt = []

file = open('data.csv', 'r')
next(file)
for line in file:
    elements = line.split(',')
    ppvt.append(int(elements[2]))
    mom_age.append(int(elements[3]))

N = 400
avg_mom_age = np.mean(mom_age)
avg_ppvt = np.mean(ppvt)

alpha_real = avg_ppvt
beta_real = avg_mom_age
eps_real = np.random.normal(0, 0.5, N)

x = np.random.normal(10, 1, N)
y_real = alpha_real + beta_real
y = y_real + eps_real

_, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(x, y)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation=0)
ax[0].plot(x, y_real, 'k')
az.plot_kde(y, ax=ax[1])
ax[1].setxlabel('y')
plt.tight_layout()

with pm.Model() as model_g:
    alpha = pm.Normal('alpha', mu = age_ppvt, sd = )
    beta = pm.Normal('beta', mu = avg_mom_age, sd = )
    epsilon = pm.HalfCauchy('epsilon', 5)
    u = pm.Deterministic('u', alpha + beta * x)
    y_pred = pm.Normal('y_pred', mu = u, sd = epsilon, observed = y)

    idata_g = pm.sample(N, tune = N, return_inferencedata = true)
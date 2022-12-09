import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import sys
import csv
import random

if __name__ == "__main__":
    #generare 500 de date:
    with open('Lab9\date.csv', 'w', newline='') as file:
        print("heeeeeeeeeeeei")
        for i in range(1,501):
            x=1.75 * np.random.random()
            y=2 + 1.5 * x
            file.writelines(str(x) + ' ' + str(y) + '\n')
    
    az.style.use('arviz-darkgrid')

    dummy_data = np.loadtxt('Lab9\date.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 3
    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()

    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')

    with pm.Model() as model_l:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        # beta = pm.Normal('beta', mu=0, sd=100)
        beta = pm.Normal('beta', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]))
        epsilon = pm.HalfNormal('epsilon', 5)
        u = alpha + beta * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=u, sd=epsilon, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        # beta = pm.Normal('beta', mu=0, sd=100, shape=order)
        # beta = pm.Normal('beta', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        beta = pm.Normal('beta', mu=0, sd=10, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        u = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=u, sd=epsilon, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_c:
        alpha = pm.Normal('alpha', mu=0, sd=1)
        # beta = pm.Normal('beta', mu=0, sd=100, shape=order)
        beta = pm.Normal('beta', mu=0, sd=10, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        u = alpha + pm.math.dot(beta, x_1s)
        y_pred = pm.Normal('y_pred', mu=u, sd=epsilon, observed=y_1s)
        idata_c = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

    alpha_l_post = idata_l.posterior['alpha'].mean(("chain", "draw")).values
    beta_l_post = idata_l.posterior['beta'].mean(("chain", "draw")).values
    y_l_post = alpha_l_post + beta_l_post * x_new

    plt.plot(x_new, y_l_post, 'C1', label='linear model')

    alpha_p_post = idata_p.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')

    alpha_c_post = idata_c.posterior['alpha'].mean(("chain", "draw")).values
    beta_c_post = idata_c.posterior['beta'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_c_post = alpha_c_post + np.dot(beta_c_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_post[idx], 'C3', label=f'cubic model order {order}')

    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()


    plt.show()

    cmp_df = az.compare({'model_l':idata_l, 'model_p':idata_p, 'model_c':idata_c}, method='BB-pseudo-BMA', ic="waic", scale="deviance")
    cmp_df2 = az.compare({'model_l':idata_l, 'model_p':idata_p, 'model_c':idata_c}, method='BB-pseudo-BMA', ic="psis_loo-cv", scale="deviance")
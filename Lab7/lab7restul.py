import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd
import math

# Ex 1:
if __name__ == "__main__":
    data = pd.read_csv('Prices.csv')

    price = data['Price'].values
    mhz = data['Speed'].values
    hd = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values

    # Ex 1:
    ex1_model = pm.Model()

    with ex1_model:
        alpha = pm.Normal('Alpha', mu = 0, sd = 10)
        beta_1 = pm.Normal('Beta_1', mu = 0, sd = 10)
        beta_2 = pm.Normal('Beta_2', mu = 0, sd = 10)
        sigma = pm.HalfCauchy('Sigma', 5)
        
        mu = pm.Deterministic('mu', alpha + beta_1 * mhz + beta_2 * np.log(hd))

        regr_like = pm.Normal('regr_like', mu = mu, sd = sigma, observed = price)
        step = pm.Slice()
        idata = pm.sample(100, return_inferencedata = True, cores = 4, step = step)

    # Ex 2:

    sig1 = az.plot_hdi(ex1_model['beta_1'], hdi_prob = 0.95)
    sig2 = az.plot_hdi(ex1_model['beta_2'], hdi_prob = 0.95)
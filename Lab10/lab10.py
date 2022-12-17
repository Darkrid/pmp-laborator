import numpy as np
import arviz as az
import pymc3 as pm
import theano as tt
from theano import tensor as tt
from theano.ifelse import ifelse

if __name__ == '__main__':
    
    clusters = [2, 3, 4]
    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means', mu=10, sd=10, shape=cluster)
            sd = pm.HalfNormal('sd', sd=10)
            order_means = pm.Potential('order_means', tt.switch(means[1] - means[0] < 0, -np.inf, 0))
            y = pm.NormalMixture('y', w=p, mu=means, sd=sd)
            idata = pm.sample(500, random_seed=123, return_inferencedata=True, init = 'adapt_diag')
            varnames = ['means']
            az.plot_trace(idata, varnames)
            az.summary(idata, varnames)

            cmp_df = az.compare(idata, method='BB-pseudo-BMA', ic="waic", scale="deviance")
            cmp_df2 = az.compare(idata, method='BB-pseudo-BMA', ic="psis_loo-cv", scale="deviance")
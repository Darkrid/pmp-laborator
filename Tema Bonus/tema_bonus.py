import pandas as pandas
import numpy as np
import pymc3 as pm
from scipy import stats

if __name__ == '__main__':
    model = pm.Model()
    with model:
        nr_clienti = pm.Poisson('nr_c', mu = 20)
        t_plasare = pm.Normal('t_p', mu = 1, sd = 0.5, shapes = 50)
        s_gatit = pm.Exponential('s_g', lam = 2, shapes = 50)
        t_client = pm.Normal('t_c', mu = 10, sd = 2, shapes = 50)
        idx = np.arange(50)
        timp = pm.Deterministic('T', pm.math.switch(nr_clienti>idx, t_plasare[idx] + s_gatit[idx], 0), shapes = 50)
        success = pm.Deterministic('S', pm.math.prod(pm.math.switch(timp < 15, 1, 0)))
        trace = pm.sample(1000)

    success = trace['S']
    case = trace['t_p']
    statii = trace['s_g']
    clienti_masa = trace['t_client']
    timp = trace['T']
    prob = len(success[(success == 1)]) / len(success)

    if prob > 0.95:
        min_case = case[(success == 1) & (timp < 15)].min()
        min_statii = statii[(success == 1) & (timp < 15)].min()

    print("Min case: ", min_case)
    print("Min statii: ", min_statii)

    if prob > 0.90:
        min_mese = clienti_masa[(success == 1)].min()

    print("Min mese: ", min_mese)
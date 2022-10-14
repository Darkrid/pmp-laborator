import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az



if __name__ == '__main__':
    # punctul 2:
    model1 = pm.Model()
    with model1:
        cutremur = pm.Bernoulli('C', 0.05)
        incendiu_p = pm.Deterministic('I_P', pm.math.switch(cutremur, 0.3, 0.1))
        incendiu = pm.Bernoulli('I', p=incendiu_p)
        alarmaIncendiu_p = pm.Deterministic('AI_P', pm.math.switch(cutremur, pm.math.switch(incendiu, 0.98, 0.2), pm.math.switch(incendiu, 0.95, 0.01)))
        alarmaIncendiu = pm.Bernoulli('AI', p=alarmaIncendiu_p, observed = 1)
        trace = pm.sample(2000)


    dictionary1 = {
        'cutremur': trace['C'].tolist(),
        'incendiu': trace['I'].tolist()
    }
    df1 = pd.DataFrame(dictionary1)

    x = df1[(df1['cutremur'] == 1)].shape[0] / df1.shape[0]

    #punctul 3:
    model2 = pm.Model()
    with model2:
        cutremur = pm.Bernoulli('C', 0.05)
        incendiu_p = pm.Deterministic('I_P', pm.math.switch(cutremur, 0.3, 0.1))
        incendiu = pm.Bernoulli('I', p=incendiu_p, observed = 1)
        alarmaIncendiu_p = pm.Deterministic('AI_P', pm.math.switch(cutremur, pm.math.switch(incendiu, 0.98, 0.2), pm.math.switch(incendiu, 0.95, 0.01)))
        alarmaIncendiu = pm.Bernoulli('AI', p=alarmaIncendiu_p, observed = 0)
        trace = pm.sample(2000)


    dictionary2 = {
        'cutremur': trace['C'].tolist(),
    }
    df2 = pd.DataFrame(dictionary2)

    y = 1 / df2.shape[0]


    print("Punctul 2:", x)
    print("Punctul 3:", y)


import numpy as np
import arviz as az
import pymc3 as pm
import matplotlib.pyplot as plt
from scipy import stats
import theano as tt
import statistics
from theano import tensor as tt
from theano.ifelse import ifelse

if __name__ == '__main__':
    # Ex1:
    def posterior_grid(grid_points = 100, heads = 28, tails = 45):
        grid = np.linspace(0, 1, grid_points)
        # prior = np.repeat(1/grid_points, grid_points)
        # prior = (grid<= 0.5).astype(int)
        prior = abs(grid - 0.5)
        likelihood = stats.binom.pmf(heads, heads + tails, grid)
        posterior = likelihood * prior
        posterior /= posterior.sum()
        return grid, posterior

    data = np.repeat([0, 1], (31, 17))
    points = 10
    h = data.sum()
    t = len(data) - h
    grid, posterior = posterior_grid(points, h, t)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'heads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ')

    # Ex2:
    error_list = []

    N = 100
    x, y = np.random.uniform (-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    error_list.append(error)

    N = 1000
    x, y = np.random.uniform (-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    error_list.append(error)
    
    N = 10000
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    error_list.append(error)

    outside = np.invert(inside)

    plt.figure(figsize=(8, 8))
    plt.plot(x[inside], y[inside], 'b.')
    plt.plot(x[outside], y[outside], 'r.')
    plt.plot(0, 0, label = f'π*= {pi:4.3f}\nerror = {error:4.3f}', alpha = 0)
    plt.axis('square')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc = 1, frameon = True, framealpha = 0.9)

    a = statistics.mean(error_list)
    b = np.std(error_list)
    plt.errorbar(a, b, fmt = 'o', color = 'black', ecolor = 'lightgray', elinewidth = 1)

    # Ex3:
    def metropolis(func, draws = 10000):
        trace = np.zeros(draws)
        old_x = 0.5
        old_prob = func.pdf(old_x)
        delta = np.random.normal(0, 0.5, draws)
        for i in range(draws):
            new_x = old_x + delta[i]
            new_prob = func.pdf(new_x)
            acceptance = new_prob / old_prob
            if acceptance >= np.random.random():
                trace[i] == new_x
                old_x = new_x
                old_prob = new_prob
            else:
                trace[i] = old_x
        return trace
    

    beta_params = [(1, 1), (20, 20), (1, 4)]

    func = stats.beta(all(list(zip(*beta_params))[0]), all(list(zip(*beta_params))[1]))
    trace = metropolis(func = func)
    x = np.linspace(0.01, .99, 100)
    y = func.pdf(x)
    plt.figure(figsize = (8, 8))
    plt.xlim(0, 1)
    plt.plot(x, y, 'C1-', lw = 3, label = 'True distribution')
    plt.hist(trace[trace > 0], bins = 25, density = True, label = 'Estimated distribution')
    plt.xlabel('x')
    plt.ylabel('pdf(x)')
    plt.yticks([])
    plt.legend()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from numpy.polynomial.polynomial import polyval, polyder
from matplotlib import cm, colorbar, colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection


def fita(AA, plot=False):
    Arange = np.arange(3, 15)

    def func(x, pars):
        a, b, c = pars
        return a * np.log(b * x) + c

    def resid(pars):
        return ((yy - func(xx, pars))**2).sum()

    def constr(pars):
        #print(np.gradient(func(xx, pars)))
        return np.gradient(func(xx, pars))

    def funct(x, pars):
        a = pars
        return polyval(x, a)

    def residt(pars):
        return ((yy - polyval(xx, pars))**2).sum()

    def constrt(pars):
        #print(polyval(xx, np.polyder(pars)))
        return polyval(xx, polyder(pars, m=2))

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    datadir = "/home/kirscher/kette_repo/AcoreNteraction/"
    data_rad = [line for line in open(datadir + 'lc4_from_rgm_3-30.dat')][1:]

    Acore = np.array([int(line.strip().split()[0]) for line in data_rad])
    acrit = np.array([float(line.strip().split()[3]) for line in data_rad])
    Rcore = [float(line.strip().split()[5]) for line in data_rad]
    anz = len(Acore)

    xx = Acore[2:]
    yy = acrit[2:]

    con1 = {'type': 'ineq', 'fun': constr}
    res = minimize(
        resid, [.3, 1., 1.],
        method='cobyla',
        options={'maxiter': 50000},
        constraints=con1)

    con1t = {'type': 'ineq', 'fun': constrt}
    rest = minimize(
        residt, [.3, 1, 5],
        method='cobyla',
        options={'maxiter': 50000},
        constraints=con1t)

    print(res)
    print(rest)

    a1 = func(AA, res.x)
    print('a(%d) = %4.4f fm^-2' % (AA, a1))
    a2 = polyval(AA, rest.x)
    print('a(%d) = %4.4f fm^-2' % (AA, a2))

    def acritf(aa):
        rr = Rcore[aa - 2] if (aa < 7) else 3.4
        return rr**(-2) * 1.5 * (aa - 1) * (aa / (3 * aa - 2))**(1.5)

    ame = acritf(AA)

    if plot:
        f = plt.figure(figsize=(10, 4))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        ax1.plot(xx, yy, 'ro', label='data')
        ax1.plot(Arange, polyval(Arange, rest.x), label='fit')
        ax1.plot(Arange, func(Arange, res.x), label='fit*')
        ax1.plot(Arange, [acritf(aa) for aa in Arange], label='Osci r2')
        ax1.legend(loc=0)
        ax2.plot(xx, constrt(rest.x), label='slope')
        ax2.legend(loc=0)
        plt.show()
        exit()
    return ame
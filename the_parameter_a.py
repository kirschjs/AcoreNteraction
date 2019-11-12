import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from numpy.polynomial.polynomial import polyval, polyder
from matplotlib import cm, colorbar, colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection


def fita(AA, order=2, orderp=1, plot=False):

    if orderp >= order:
        print(
            'purely positive polynomial selected, so you are deprived of\n all the \"negative benefits!\" '
        )
        ordern = 0
    else:
        ordern = order - orderp

    print('order of all powers          = %d' % order)
    print('order of the negative powers = %d' % ordern)
    print('order of the positive powers = %d' % orderp)
    Arange = np.array(AA).astype(float)

    def func(x, pars):
        a, b, c = pars
        return a * np.log(b * x) + c

    def resid(pars):
        return ((yy - func(xx, pars))**2).sum()

    def constr(pars):
        return np.gradient(func(xx, pars))

    def funct(x, pars):
        a = pars
        a[1] = 0
        if orderp > 0:
            a[ordern + 1] = 0
            return a[0] + x**(-ordern - 1) * polyval(
                x, a[1:ordern + 1]) + polyval(x, a[ordern + 1:])
        else:
            return a[0] + x**(-ordern - 1) * polyval(x, a[1:ordern + 1])

    def residnp(pars):
        return ((yy - funct(xx, pars))**2).sum()

    def constrnp(pars):
        if orderp > 0:
            return (-1) * Arange**(-ordern - 1) * polyval(
                Arange, polyder(pars[1:ordern + 1], m=1)) + polyval(
                    Arange, polyder(pars[ordern + 1:], m=1))
        else:
            return (-1) * Arange**(-ordern - 1) * polyval(
                Arange, polyder(pars[1:ordern + 1], m=1))

    def residt(pars):
        return ((yy - polyval(xx, pars))**2).sum()

    def constrt(pars):
        return polyval(Arange, polyder(pars, m=1))

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

    xx = np.array(Acore).astype(float)
    yy = np.array(acrit).astype(float)

    con1np = {'type': 'ineq', 'fun': constrnp}
    resnp = minimize(
        residnp,
        np.random.rand(order + 1),
        method='cobyla',
        options={'maxiter': 50000},
        constraints=con1np)

    print(resnp)

    con1t = {'type': 'ineq', 'fun': constrt}
    rest = minimize(
        residt,
        np.random.rand(order),
        method='cobyla',
        options={'maxiter': 50000},
        constraints=con1t)

    apoly = polyval(Arange, rest.x)
    apolyn = funct(Arange, resnp.x)

    heuriger = 1. / 2.

    def acritf(aa):

        zet = Rcore[-1] / Acore[-1]**(1. / 3.)
        rr = Rcore[aa - 2] if (aa < 7) else zet * aa**(1. / 3.)
        # return result in volume approximation for core numbers for which no
        # microscopic data is available
        a = heuriger * rr**(-2) * 1.5 * (aa - 1)**2 / aa if (
            aa > 6) else acrit[aa - 2]

        return a

    ame = [acritf(int(aa)) for aa in Arange]

    f = plt.figure(figsize=(10, 4))
    f.suptitle(
        r'$\kappa=\langle rms_{A=6}\rangle\cdot 6^{-1/3}=%4.2f$' %
        (Rcore[-1] / Acore[-1]**(1. / 3.)),
        fontsize=14)
    ax1 = f.add_subplot(111)
    ax1.set_xlabel(r'$A$', fontsize=12)
    ax1.set_ylabel(r'core oscillator $a [fm^{-2}]$', fontsize=12)
    #ax2 = f.add_subplot(122)
    #ax1.set_ylim( [np.min(funct(Arange, resnp.x)),          np.max(funct(Arange, resnp.x))])
    #ax1.plot(Arange, funct(Arange, resnp.x), label=r'$f(x)=\sum p_ix^i+n_ix^{-i}$')
    ax1.plot(xx, yy, 'ro', label=r'SVM data')
    #        ax1.plot(Arange, polyval(Arange, rest.x), label=r'$f(x)=\sum p_ix^i$')
    ax1.plot(
        Arange, [acritf(int(aa)) for aa in Arange],
        label=r'$\kappa\cdot\frac{3\,(A-1)^2}{2\,A^{5/3}}$')
    ax1.legend(loc=0, fontsize=14)
    #ax2.plot(Arange, constrt(rest.x), label=r'slope$+$')
    #ax2.plot(Arange, constrt(resnp.x), label=r'slope$\pm$')
    #ax2.legend(loc=0)
    strFile = 'a_para_fit.pdf'
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    if plot:
        plt.show()
    plt.clf()

    return apolyn, apoly, ame
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

datadir = "/home/kirscher/kette_repo/AcoreNteraction/"


def polinv(x, a, b, c, d):
    return a + b / x + c / x**2 + d / x**3


def pol(x, a, b, c, d):
    return a + b * x + c * x**2 + d * x**3


def pol4(x, a, b, c, d, e):
    return a + b * x + c * x**2 + d * x**3 + e * x**4


data_rad = [line for line in open(datadir + 'rad_core_3.dat')][1:]
anz = len(data_rad[0].strip().split()) - 1
X = np.array([float(line.strip().split()[0]) for line in data_rad])
Y = np.array([float(line.strip().split()[7 - AA]) for line in data_rad])
fun = pol4
mask = np.isfinite(Y)
if not speculative:
    f2 = interp1d(X[mask], Y[mask], kind='cubic')
else:
    popt, pcov = curve_fit(fun, X[mask], Y[mask])
    ics = np.arange(rang[0], rang[1], rang[0])
    if fun == pol4:
        why = fun(ics, popt[0], popt[1], popt[2], popt[3], popt[4])
    else:
        why = fun(ics, popt[0], popt[1], popt[2], popt[3])
    f2 = interp1d(ics, why, kind='linear')
#plt.plot(X[mask],Y[mask],'ko',label="data")
return f2(lamb)

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import timeit
import scipy
from scipy.special import eval_genlaguerre, iv, spherical_jn
from multiprocessing import Lock, Process, Queue, current_process, Pool, cpu_count
from potcoeffs import *
from LECs_interpolation_constr import *
from the_parameter_a import *
from sympy import lambdify
from sympy.abc import x
from sympy.physics.quantum.cg import CG, Wigner3j
import numpy.ma as ma

# what is done and (!) why?
now = datetime.now()
secundo = int(str(now).split(':')[1])


def log_d_fact(n):
    if n <= 0:
        return 0
    else:
        return np.log(n) + log_d_fact(n - 2)


def log_fact(n):
    if n <= 0:
        return 0
    else:
        return np.log(n) + log_fact(n - 1)


def psi(r, n, l, nu):
    N = np.sqrt(
        np.sqrt(2. * (nu**3) / np.pi) *
        (2.**(n + 2 * l + 3
              ) * nu**l * np.exp(log_fact(n) - log_d_fact(2 * n + 2 * l + 1))))
    psi = r * N * r**l * np.exp(-nu * r**2) * eval_genlaguerre(
        n, l + 0.5, 2 * nu * r**2)
    return psi


Ncore = 6

NState = 4
Rmax = 35
order = 200

Lrel = 1  # Momento angolare

# physical system
mpi = '137'
m = 938.12
hbar = 197.327

x, w = np.polynomial.legendre.leggauss(order)
# Translate x values from the interval [-1, 1] to [a, b]
a = 0.0
b = Rmax
t = 0.5 * (x + 1) * (b - a) + a
gauss_scale = 0.5 * (b - a)

psiRN = np.zeros((order, NState))

mu = Ncore * m / (Ncore + 1.)
mh2 = hbar**2 / (2 * mu)

omega = 0.1
nu = mu * omega / (2 * hbar)

for y in range(NState):
    for x in range(order):
        psiRN[x, y] = psi(t[x], y, Lrel, nu)

f = plt.figure(figsize=(22, 16))
#f.suptitle(r'', fontsize=14)
ax2 = f.add_subplot(111)

ax2.set_xlabel(r'$r$ [fm]', fontsize=44, labelpad=20)

ax2.set_ylabel(r'$\psi(r)$ [.]', fontsize=44, labelpad=20)

plt.xticks(fontsize=40, rotation=0)
plt.yticks(fontsize=40, rotation=0)
ax2.tick_params(axis='both', which='major', pad=15)

[
    ax2.plot(t, psiRN[:, nst], markersize=22, label=r'$n=%d$' % (nst))
    for nst in range(NState)
]

#ax1.set_ylim([0, 5])
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

plt.legend(loc='best', numpoints=1, fontsize=40)

strFile = 'HO_wavefunctions.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
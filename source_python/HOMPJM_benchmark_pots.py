import numpy as np
import matplotlib.pyplot as plt
import sys
import timeit
import scipy
from scipy.special import eval_genlaguerre, iv, spherical_jn
from potcoeffs import *
from LECs_interpolation import *
from the_parameter_a import *
from sympy import lambdify
from sympy.abc import x
import numpy.ma as ma

from kette.math_util import log_fact


def log_d_fact(n):
    """Calculates `ln(n!!)` by adding `ln(n) + ln(n-2) + ln(n-4) + ...`
    """
    result = 0
    while n > 2:
        result += np.log(n)
        n -= 2
    return result


def psi(r, n, l, nu):
    N = np.sqrt(
        np.sqrt(2. * (nu**3) / np.pi) *
        (2.**(n + 2 * l + 3
              ) * nu**l * np.exp(log_fact(n) - log_d_fact(2 * n + 2 * l + 1))))
    psi = r * N * r**l * np.exp(-nu * r**2) * eval_genlaguerre(
        n, l + 0.5, 2 * nu * r**2)
    return psi


def ddpsi(r, n, l, nu):
    N = np.sqrt(
        np.sqrt(2. * (nu**3) / np.pi) *
        (2.**(n + 2 * l + 3
              ) * nu**l * np.exp(log_fact(n) - log_d_fact(2 * n + 2 * l + 1))))
    ddpsi = np.exp(-r**2 * nu) * (
        16. * r**
        (4 + l - 1) * nu**2 * eval_genlaguerre(n - 2, l + 2.5, 2 * nu * r**2) +
        4. * r**(2 + l - 1) * nu *
        (-3 - 2 * l + 4 * r**2 * nu
         ) * eval_genlaguerre(n - 1, l + 1.5, 2 * nu * r**2) + r**(l - 1) *
        (l * (l + 1) - 2 * (3 + 2 * l) * r**2 * nu + 4 * r**4 * nu**2
         ) * eval_genlaguerre(n, l + 0.5, 2 * nu * r**2))
    return N * ddpsi


states_print = 3  # How many energies do you want?

Sysem = "Hiyama_lambda_alpha"

if Sysem == "HObenchmark":
    # HO  benchmark
    m = 1
    mu = m
    hbar = 1
    omegas = [1.]
    L = 1
    potargs = [MuOmegaSquare]

    def pot_local(r, argv):
        return 0.5 * MuOmegaSquare * r**2

    interaction = "Local"
    energydepen = False

elif Sysem == "Hiyama_lambda_alpha":
    # Hyama non local benchmark
    # E = -3.12 MeV
    NState = 30  #Number of basys states
    Rmax = 15
    order = 200
    m_alpha = 3727.379378
    m_lambda = 1115.683
    mu = (m_alpha * m_lambda) / (m_alpha + m_lambda)
    hbar = 197.327
    mh2 = hbar**2 / (2 * mu)
    omegas = np.linspace(0.001, 0.04, 25)

    potargs = []
    L = 0
    interaction = "NonLocal"

    #interaction = "Local"


    def pot_nonlocal(rl, rr, argv):
        z = 1j
        # Hiyama's parameters which must yield E_0(L=0)=3.12MeV
        #        C2      -alpha  -beta   -gamma
        #        Ui      -(g+d)  2(g-d)   -(g+d)
        v2H = [-0.3706, -0.5821, -0.441, -0.5821]
        v3H = [-12.94, -1.1441, -1.565, -1.1441]
        v4H = [-331.2, -3.1108, -5.498, -3.1108]

        vv = 0.
        for v in [v2H, v3H, v4H]:
            vv += (1j**L) * (v[0] * np.nan_to_num(
                spherical_jn(L, 1j * v[2] * rr * rl),
                nan=0.0,
                posinf=0.0,
                neginf=0.0) * np.exp(v[1] * rr**2 + v[3] * rl**2))

        return np.nan_to_num(vv.real)

    def pot_local(r, argv):
        Vvv = (-17.49) * np.exp(-0.2752 * (r**2)) + (
            -127.0) * np.exp(-0.4559 *
                             (r**2)) + (497.8) * np.exp(-0.6123 * (r**2))
        return Vvv

else:
    print("ERROR: I do not know the sysem you want")
    quit()

if __name__ == '__main__':

    x, w = np.polynomial.legendre.leggauss(order)
    # Translate x values from the interval [-1, 1] to [a, b]
    a = 0.0
    b = Rmax
    t = 0.5 * (x + 1) * (b - a) + a
    gauss_scale = 0.5 * (b - a)

    egs = []

    for omega in omegas:

        H = np.zeros((NState, NState))
        Kin = np.zeros((NState, NState))
        Vnonloc = np.zeros((NState, NState))
        Vloc = np.zeros((NState, NState))
        U = np.zeros((NState, NState))
        nu = mu * omega / (2 * hbar)

        start_time = timeit.default_timer()
        start_time2 = start_time

        psiRN = np.zeros((order, NState))
        ddpsiRN = np.zeros([order, NState])
        VlocRN = np.zeros(order)

        for y in range(NState):
            for x in range(order):
                psiRN[x, y] = psi(t[x], y, L, nu)
                ddpsiRN[x, y] = ddpsi(t[x], y, L, nu)

        start_time = timeit.default_timer()
        VlocRN[:] = pot_local(t[:], potargs) + mh2 * L * (L + 1) / t[:]**2

        if (interaction == "NonLocal"):

            VnolRN = np.fromfunction(
                lambda x, y: pot_nonlocal(t[x], t[y], potargs), (order, order),
                dtype=int)

        start_time = timeit.default_timer()

        for i in np.arange(NState):
            for j in np.arange(i + 1):
                U[i][j] = np.sum(
                    psiRN[:, i] * psiRN[:, j] * w[:]) * gauss_scale
                Kin[i][j] = np.sum(
                    psiRN[:, i] * ddpsiRN[:, j] * w[:]) * gauss_scale
                Vloc[i][j] = np.sum(
                    psiRN[:, i] * VlocRN[:] * psiRN[:, j] * w[:]) * gauss_scale
                if (interaction == "NonLocal"):
                    for k in range(order):
                        Vnonloc[i][j] = Vnonloc[i][j] + 4. * np.pi * np.sum(
                            t[:] * VnolRN[k, :] * psiRN[:, j] * w[:]
                        ) * psiRN[k, i] * t[k] * w[k] * gauss_scale**2
                H[j][i] = H[i][j]
                Vnonloc[j][i] = Vnonloc[i][j]
                Vloc[j][i] = Vloc[i][j]
                Kin[j][i] = Kin[i][j]
                U[j][i] = U[i][j]

        Kin = -mh2 * Kin
        H = Vloc + Vnonloc + Kin

        # Check if basis orthonormal:
        if np.sum(abs(np.eye(NState) - U)) > 0.1 * NState**2:
            print(" ")
            print("WARNING: omega = ", omega)
            print("   >>  basis not sufficiently orthonormal: ")
            print("   >>  average deviation from unit matrix:",
                  np.round(np.sum(abs(np.eye(NState) - U)) / NState**2, 2))
            #print(np.round(U,2))
            print(" ")
            continue

        for i in np.arange(1, NState + 1):

            #valn, vecn = scipy.linalg.eig(H[:i, :i])
            #zn = np.argsort(valn)
            #zn = zn[0:states_print]
            #energiesn = (valn[zn])
            energiesn = scipy.linalg.eigvalsh(H[:i, :i])

        print("Nstates = %3d Omega = %4.4f  Re[E_GS] = %4.4f dT = %5.5f" %
              (i, omega, np.real(energiesn[0]),
               timeit.default_timer() - start_time))

        egs.append(np.real(energiesn[0]))

f = plt.figure(figsize=(18, 12))
f.suptitle(r'', fontsize=14)

ax1 = f.add_subplot(111)

ax1.set_xlabel(r'$\omega$ [fm$^{-2}$]', fontsize=12)
ax1.set_ylabel(r'$E_0$ [MeV]', fontsize=12)

ax1.plot(omegas, egs, label=r'Hiyama\'s')

plt.legend(loc='best', fontsize=24)

strFile = 'testHiyama.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
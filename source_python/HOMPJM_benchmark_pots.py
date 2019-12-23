import numpy as np
import matplotlib.pyplot as plt
import sys
import timeit
import scipy
from scipy.special import eval_genlaguerre, iv, spherical_jn
from multiprocessing import Lock, Process, Queue, current_process, Pool, cpu_count
from potcoeffs import *
from LECs_interpolation import *
from the_parameter_a import *
from sympy import lambdify
from sympy.abc import x
from sympy.physics.quantum.cg import CG, Wigner3j
import numpy.ma as ma

###############################
######## V LO pionless ########
###############################
lambdas = [
    0.05, 0.1, 0.16, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
    0.75, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.5, 2, 3, 4, 6, 10
]
LeCofL = [
    -1.552, -2.242, -3.440, -4.040, -6.397, -7.808, -9.310, -10.976, -12.782,
    -14.726, -16.809, -19.032, -21.394, -23.894, -26.536, -32.234, -35.291,
    -38.489, -41.824, -45.299, -52.666, -78.111, -131.646, -280.457, -484.921,
    -1060.797, -2880.387
]
LeDofL = [
    -0.324, -0.246, 0.019, 0.223, 1.207, 1.981, 2.828, 3.913, 5.209, 6.718,
    8.468, 10.463, 12.734, 15.276, 18.137, 24.813, 28.664, 32.911, 37.527,
    42.570, 53.971, 100.476, 232.413, 854.107, 2495.364, 16915.630, 756723.220
]

###############################
######## Wave function ########
###############################


def log_d_fact(n):
    """Calculates `ln(n!!)` by adding `ln(n) + ln(n-2) + ln(n-4) + ...`
    """
    if n <= 0:
        return 0

    return np.log(n) + log_d_fact(n - 2)


def log_fact(n):
    """Calculates `ln(n!)` by adding `ln(n) + ln(n-1) + ...`
    """
    # TODO: Consider benchmarking vs naive implementation of `np.log(...)`.
    if n < 0:
        raise ValueError('Factorial not defined for negative numbers')
    if not isinstance(n, int):
        raise ValueError('Factorial only defined for integral values')

    result = 0
    while n > 0:
        result += np.log(n)
        n -= 1

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


#######################
### general options ###
#######################
pedantic = True
states_print = 3  # How many energies do you want?

# Parallel of this version not implemented
parallel = False  # Do you want trallel version?
Nprocessors = 6  # Number of processors

Sysem = "Hiyama_lambda_alpha"

############################
### Potential definition ###
############################
if Sysem == "Pionless":
    # Physical system and benchmark
    NState = 25  #Number of basys states
    order = 500  # Integration order
    m = 938.858
    mu = m / 2.
    hbar = 197.327
    mh2 = hbar**2 / (2 * mu)
    Rmax = 20  #Max R integration
    omegas = [1.0]
    L = 0
    potargs = []

    def pot_local(r, argv):
        return -505.1703491 * np.exp(-4. * r**2)

    interaction = "Local"
    energydepen = True

elif Sysem == "HObenchmark":
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
    NState = 28  #Number of basys states
    Rmax = 25
    order = 500
    m_alpha = 3727.379378
    m_lambda = 1115.683
    mu = (m_alpha * m_lambda) / (m_alpha + m_lambda)
    hbar = 197.327
    mh2 = hbar**2 / (2 * mu)
    omegas = np.linspace(0.05, 0.4, 5)

    potargs = []
    L = 0
    interaction = "NonLocal"
    #interaction = "Local"
    energydepen = False

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
    if parallel:
        p = Pool(Nprocessors)
        print("Number of CPU: ", cpu_count())

    if pedantic:
        print("")
        print("parallel: ", parallel)
        print("--- " + Sysem + "---")
        print("# of states : " + str(NState))
        print("Max R       : " + str(Rmax))
        print("Gauss order : " + str(order))
        print("Mass        : " + str(mu))
        print("hbar        : " + str(hbar))
        print("h^2/2m      : " + str(np.round(mh2, 3)))
        print("")

    x, w = np.polynomial.legendre.leggauss(order)
    # Translate x values from the interval [-1, 1] to [a, b]
    a = 0.0
    b = Rmax
    t = 0.5 * (x + 1) * (b - a) + a
    gauss_scale = 0.5 * (b - a)

    ene_omega_Kex = []
    ene_omega_Kdi = []
    for omega in omegas:

        ###########################
        ### All vectors to zero ###
        ###########################
        H = np.zeros((NState, NState))
        Kin = np.zeros((NState, NState))
        Vnonloc = np.zeros((NState, NState))
        Vloc = np.zeros((NState, NState))
        U = np.zeros((NState, NState))
        Uex = np.zeros((NState, NState))
        nu = mu * omega / (2 * hbar)
        if (pedantic): print("Omega       : " + str(omega))
        if (pedantic): print("nu          : " + str(np.round(nu, 3)))
        if (pedantic): print(" ")

        if (pedantic): print("Creation of integration array: ")
        start_time = timeit.default_timer()
        start_time2 = start_time

        psiRN = np.zeros((order, NState))
        ddpsiRN = np.zeros([order, NState])
        VlocRN = np.zeros(order)
        VexRN = np.zeros((order, order))

        for y in range(NState):
            for x in range(order):
                psiRN[x, y] = psi(t[x], y, L, nu)
                ddpsiRN[x, y] = ddpsi(t[x], y, L, nu)

        if (pedantic):
            print(" >> Wave function: (", timeit.default_timer() - start_time,
                  " s )")
        start_time = timeit.default_timer()
        VlocRN[:] = pot_local(t[:], potargs) + mh2 * L * (L + 1) / t[:]**2
        if (pedantic):
            print(" >> Local potential:  (",
                  timeit.default_timer() - start_time, " s )")
        start_time = timeit.default_timer()

        if (interaction == "NonLocal"):
            if energydepen:
                VexRN = np.fromfunction(
                    lambda x, y: exchange_kernel(t[x], t[y], potargs),
                    (order, order),
                    dtype=int)
            VnolRN = np.fromfunction(
                lambda x, y: pot_nonlocal(t[x], t[y], potargs), (order, order),
                dtype=int)
            #print("Vnonlocal(0.33 , 0.75)= ",pot_nonlocal(0.33 , 0.75,potargs) )
            #stop()
            if (pedantic):
                print(" >> NonLocal potential:  (",
                      timeit.default_timer() - start_time, " s )")
            start_time = timeit.default_timer()

        if (pedantic):
            print("Array creation time:", timeit.default_timer() - start_time2,
                  " s")
        start_time = timeit.default_timer()

        if (pedantic): print("Array integration:")

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
                        Uex[i][j] = Uex[i][j] + 4. * np.pi * np.sum(
                            t[:] * VexRN[k, :] * psiRN[:, j] * w[:]
                        ) * psiRN[k, i] * t[k] * w[k] * gauss_scale**2
                    for k in range(order):
                        #Vnonloc[i][j] = Vnonloc[i][j] +
                        #np.sum(VnolRN[k, :] * psiRN[:, j] * w[:]) * psiRN[k, i] * w[k] * gauss_scale**2
                        Vnonloc[i][j] = Vnonloc[i][j] + 4. * np.pi * np.sum(
                            t[:] * VnolRN[k, :] * psiRN[:, j] * w[:]
                        ) * psiRN[k, i] * t[k] * w[k] * gauss_scale**2

        if (pedantic):
            print("Integration time:", timeit.default_timer() - start_time,
                  " s")
        start_time = timeit.default_timer()

        Kin = -mh2 * Kin
        H = Vloc + Vnonloc + Kin

        for i in np.arange(NState):
            for j in np.arange(0, i):
                H[j][i] = H[i][j]
                Vnonloc[j][i] = Vnonloc[i][j]
                Vloc[j][i] = Vloc[i][j]
                Kin[j][i] = Kin[i][j]
                U[j][i] = U[i][j]

        # Check if basis orthonormal:
        if np.sum(abs(np.eye(NState) - U)) > 0.1 * NState**2:
            print(" ")
            print("WARNING: omega = ", omega)
            print("   >>  basis not sufficiently orthonormal: ")
            print("   >>  average deviation from unit matrix:",
                  np.round(np.sum(abs(np.eye(NState) - U)) / NState**2, 2))
            #print(np.round(U,2))
            print("--------------------------")
            print(" ")
            print(" ")
            print(" ")
            continue

        debug = False
        if debug:
            print('condition number of H:', np.linalg.cond(H))
            print("Gauss scale: ", gauss_scale)
            print(" ")
            print("U: ")
            print(np.round(U, 2))
            print("Vloc: ")
            print(np.around(Vloc, 2))
            print("Vnonloc:  ")
            print(np.around(Vnonloc, 2))
            print("Kin: ")
            print(np.around(Kin))
            print("H: ")
            print(np.around(H))
            #print("Vloc/Vnonloc: ")
            #print(np.around(Vloc / Vnonloc, 3))

            np.savetxt('Unit_loc.txt', np.matrix(U), fmt='%12.4f')
            np.savetxt('V_loc.txt', np.matrix(Vloc), fmt='%12.4f')
            np.savetxt('V_nonloc.txt', np.matrix(Vnonloc), fmt='%12.4f')
            np.savetxt('E_kin.txt', np.matrix(Kin), fmt='%12.4f')
            np.savetxt('Hamiltonian.txt', np.matrix(H), fmt='%12.4f')
            exit()

        # Diagonalize with Kex, i.e., solve gen. EV
        if (pedantic):
            print(" ")
            for i in np.arange(1, NState + 1):
                #for i in np.arange(NState+1):
                valn, vecn = scipy.linalg.eig(H[:i, :i])
                valg, vecg = scipy.linalg.eig(H[:i, :i], Kex[:i, :i])
                zg = np.argsort(valg)
                zg = zg[0:states_print]
                zn = np.argsort(valn)
                zn = zn[0:states_print]
                energiesn = (valg[zn])
                energiesg = (valg[zg])
                valn, vecn = scipy.linalg.eig(H[:i, :i])
                zn = np.argsort(valn)
                zn = zn[0:states_print]
                energiesn = (valn[zn])
        else:
            for i in np.arange(NState, NState + 1):
                #for i in np.arange(NState+1):
                #valn, vecn = scipy.linalg.eig(H[:i, :i], np.eye(NState))
                valg, vecg = scipy.linalg.eig(H[:i, :i], Kex[:i, :i])
                zg = np.argsort(valg)
                zg = zg[0:states_print]
                energiesg = (valg[zg])

                if (pedantic):
                    print("states: " + str(i) + "  Energies(g): " +
                          str(energiesg) + "  Energies(n): " + str(energiesn))

        if (pedantic):
            print("Diagonalization time:", timeit.default_timer() - start_time,
                  " s")

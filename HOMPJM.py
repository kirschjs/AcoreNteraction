import numpy as np
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
pedantic = 0
states_print = 3  # How many energies do you want?

# Parallel of this version not implemented
parallel = False  # Do you want trallel version?
Nprocessors = 6  # Number of processors

mpi = '137'
m = 938.12
hbar = 197.327
# Prague-Jerusalem-Manchester effective A-1 interaction

# number of core particles (capital A in docu)
Lc = []

for Ncore in range(16, 20):

    Lamb = Lc[-1][1] if Lc != [] else 1.0
    Lamb = 10.0
    stable = True
    while (stable):
        NState = 20
        Rmax = 20
        order = 300
        omegas = np.linspace(0.01, 0.5, 10)
        #omegas = [1.5]

        L = 1

        parametriz = "C1_D4"
        LeC = return_val(Lamb, parametriz, "C")
        LeD = return_val(Lamb, parametriz, "D")

        coreosci = fita(Ncore, plot=True)

        mu = Ncore * m / (Ncore + 1.)
        mh2 = hbar**2 / (2 * mu)

        potargs = [coreosci, Ncore, float(Lamb), LeC, LeD]

        interaction = "NonLocal"
        energydepen = False

        aa1 = alf1(potargs)
        aa2 = alf2(potargs)
        aa3 = alf3(potargs)
        aa4 = alf4(potargs)
        bb1 = bet1(potargs)
        bb2 = bet2(potargs)
        bb3 = bet3(potargs)
        bb4 = bet4(potargs)
        gg1 = gam1(potargs)
        gg2 = gam2(potargs)
        gg3 = gam3(potargs)
        gg4 = gam4(potargs)
        zz1 = zeta1(potargs)
        zz2 = zeta2(potargs)
        zz3 = zeta3(potargs)
        zz4 = zeta4(potargs)

        nn1 = eta1(potargs)
        nn2 = eta2(potargs)
        nn3 = eta3(potargs)
        kk1 = kappa1(potargs)
        kk2 = kappa2(potargs)
        kk3 = kappa3(potargs)

        # this might overdo it but the sympy expressions
        # were handled erroneously before
        w3j_m2 = (Wigner3j(1, L - 1, L, 0, 0, 0).doit())**2
        w3j_p2 = (Wigner3j(1, L + 1, L, 0, 0, 0).doit())**2
        wigm = 0 if (w3j_m2 == 0) else float(w3j_m2.evalf())
        wigp = 0 if (w3j_p2 == 0) else float(w3j_p2.evalf())

        print(" - RGM potential -")
        print("Lambda  = " + str(Lamb))
        print("Ncore   = " + str(Ncore))
        print("a core  = " + str(coreosci))
        if pedantic:
            print("LEC 2b  = " + str(LeC))
            print("LEC 3b  = " + str(LeD))
            print(" -- ")
            print("alpha1  = " + str(aa1))
            print("alpha2  = " + str(aa2))
            print("alpha3  = " + str(aa3))
            print("alpha4  = " + str(aa4))
            print("beta1   = " + str(bb1))
            print("beta2   = " + str(bb2))
            print("beta3   = " + str(bb3))
            print("beta4   = " + str(bb4))
            print("gamma1  = " + str(gg1))
            print("gamma2  = " + str(gg2))
            print("gamma3  = " + str(gg3))
            print("gamma4  = " + str(gg4))
            print("zeta1   = " + str(zz1))
            print("zeta2   = " + str(zz2))
            print("zeta3   = " + str(zz3))
            print("zeta4   = " + str(zz4))
            print("eta1    = " + str(nn1))
            print("eta2    = " + str(nn2))
            print("eta3    = " + str(nn3))
            print("kappa1  = " + str(kk1))
            print("kappa2  = " + str(kk2))
            print("kappa3  = " + str(kk3))
            print(" -- ")
            print("W3J\'s  = " + str(w3j_p2) + "  " + str(w3j_m2))
            print(" -- ")
            exit()

        def exchange_kernel(rl, rr, argv):

            rr2 = rr**2
            rl2 = rl**2

            # (-) b/c this term goes to the RHS of Eq.(12)
            V1ex = -(1j**L) * ((zz1 * np.nan_to_num(
                spherical_jn(L, 1j * bb1 * rr * rl),
                nan=0.0,
                posinf=0.0,
                neginf=0.0) * np.exp(-aa1 * rr2 - gg1 * rl2)))

            return np.nan_to_num(np.real(V1ex))

        def pot_nonlocal(rl, rr, argv):

            rr2 = rr**2
            rl2 = rl**2

            V1 = -mh2 * zz1 * np.exp(-aa1 * rr2 - gg1 * rl2) * (
                4. * aa1 * bb1 * rl * rr *
                ((1j**
                  (L - 1) * np.nan_to_num(
                      spherical_jn(L - 1, 1j * bb1 * rr * rl),
                      nan=0.0,
                      posinf=0.0,
                      neginf=0.0) * wigm * (2 * L - 3) + 1j**
                  (L + 1) * np.nan_to_num(
                      spherical_jn(L + 1, 1j * bb1 * rr * rl),
                      nan=0.0,
                      posinf=0.0,
                      neginf=0.0) * wigp * (2 * L - 1)) +
                 (-4. * aa1**2 * rr2 + 2 * aa1 - bb1**2 * rl2 + L *
                  (L + 1) / rr2) * 1j**L * np.nan_to_num(
                      spherical_jn(L, 1j * bb1 * rr * rl),
                      nan=0.0,
                      posinf=0.0,
                      neginf=0.0)))

            # this is simple (still missing (4 pi i^l ) RR')
            V234 = -(1j**L) * (
                (zz2 * np.nan_to_num(
                    spherical_jn(L, 1j * bb2 * rr * rl),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0) * np.exp(-aa2 * rr2 - gg2 * rl2)) +
                (zz3 * np.nan_to_num(
                    spherical_jn(L, 1j * bb3 * rr * rl),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0) * np.exp(-aa3 * rr2 - gg3 * rl2)) +
                (zz4 * np.nan_to_num(
                    spherical_jn(L, 1j * bb4 * rr * rl),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0) * np.exp(-aa4 * rr2 - gg4 * rl2)))

            # this function is high unstable for large r (it gives NaN but it should give 0.)
            return np.nan_to_num(np.real(-V1 - V234))

        def pot_local(r, argv):
            r2 = r**2
            VnnDI = nn1 * np.exp(-kk1 * r2)
            VnnnDIarm = nn2 * np.exp(-kk2 * r2)
            VnnnDIstar = nn3 * np.exp(-kk3 * r2)
            return VnnDI + VnnnDIarm + VnnnDIstar

        x, w = np.polynomial.legendre.leggauss(order)
        # Translate x values from the interval [-1, 1] to [a, b]
        a = 0.0
        b = Rmax
        t = 0.5 * (x + 1) * (b - a) + a
        gauss_scale = 0.5 * (b - a)

        ene_omega_Kex = []
        ene_omega_Kdi = []

        stable = False

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
                print(" >> Wave function: (",
                      timeit.default_timer() - start_time, " s )")
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
                    lambda x, y: pot_nonlocal(t[x], t[y], potargs),
                    (order, order),
                    dtype=int)
                #print("Vnonlocal(0.33 , 0.75)= ",pot_nonlocal(0.33 , 0.75,potargs) )
                #stop()
                if (pedantic):
                    print(" >> NonLocal potential:  (",
                          timeit.default_timer() - start_time, " s )")
                start_time = timeit.default_timer()

            if (pedantic):
                print("Array creation time:",
                      timeit.default_timer() - start_time2, " s")
            start_time = timeit.default_timer()

            if (pedantic): print("Array integration:")

            for i in np.arange(NState):
                for j in np.arange(i + 1):
                    U[i][j] = np.sum(
                        psiRN[:, i] * psiRN[:, j] * w[:]) * gauss_scale
                    Kin[i][j] = np.sum(
                        psiRN[:, i] * ddpsiRN[:, j] * w[:]) * gauss_scale
                    Vloc[i][j] = np.sum(psiRN[:, i] * VlocRN[:] * psiRN[:, j] *
                                        w[:]) * gauss_scale
                    if (interaction == "NonLocal"):
                        for k in range(order):
                            Uex[i][j] = Uex[i][j] + 4. * np.pi * np.sum(
                                t[:] * VexRN[k, :] * psiRN[:, j] * w[:]
                            ) * psiRN[k, i] * t[k] * w[k] * gauss_scale**2
                        for k in range(order):
                            #Vnonloc[i][j] = Vnonloc[i][j] +
                            #np.sum(VnolRN[k, :] * psiRN[:, j] * w[:]) * psiRN[k, i] * w[k] * gauss_scale**2
                            Vnonloc[i][
                                j] = Vnonloc[i][j] + 4. * np.pi * np.sum(
                                    t[:] * VnolRN[k, :] * psiRN[:, j] * w[:]
                                ) * psiRN[k, i] * t[k] * w[k] * gauss_scale**2

            if (pedantic):
                print("Integration time:",
                      timeit.default_timer() - start_time, " s")
            start_time = timeit.default_timer()

            Kin = -mh2 * Kin
            H = Vloc + Vnonloc + Kin
            #H = Vnonloc + Kin
            Kex = U + Uex if energydepen else U

            for i in np.arange(NState):
                for j in np.arange(0, i):
                    H[j][i] = H[i][j]
                    Vnonloc[j][i] = Vnonloc[i][j]
                    Vloc[j][i] = Vloc[i][j]
                    Kin[j][i] = Kin[i][j]
                    U[j][i] = U[i][j]
                    Kex[j][i] = Kex[i][j]

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

            for i in np.arange(NState, NState + 1):
                #for i in np.arange(NState+1):
                #valn, vecn = scipy.linalg.eig(H[:i, :i], np.eye(NState))
                valg, vecg = scipy.linalg.eig(H[:i, :i], Kex[:i, :i])
                zg = np.argsort(valg)
                zg = zg[0:states_print]
                energiesg = (valg[zg])

            ene_omega_Kex.append(energiesg[0])
            print("nu: " + str(np.round(nu, 5)) + "  states: " + str(i) +
                  "  Energies(g): " + str(energiesg))
            if energiesg[0] < 0:
                print('  A  Lambda        a    B(A+1)\n%3d %7.2f %8.4f %9.4f' %
                      (Ncore, Lamb, coreosci, np.real(energiesg[0])))
                stable = True
        if stable != True:
            Lc.append([Ncore, Lamb])
            continue
        else:
            Lamb *= 1.2

    print(Lc)
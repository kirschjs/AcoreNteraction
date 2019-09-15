import numpy as np
from scipy.special import eval_genlaguerre, iv, spherical_jn
import matplotlib.pyplot as plt
import sys
from multiprocessing import Lock, Process, Queue, current_process, Pool, cpu_count
import timeit

###############################
######## V LO pionless ########
###############################
lec_list_c = {}
lec_list_c['137'] = {  #d0GS TNI-UIX  ZENTRAL NNN   PROJ           d0ES
    '2.00':
    [-142.364, -106.2793, 68.488, 66.8240, 33.4723, -276.2272, -0.830307],
    '4.00':
    [-505.164, -434.9584, 677.799, 724.4771, 338.7663, -892.3704, -7.645753],
    '6.00': [
        -1090.584, -986.2518, 2652.651, 3198.9831, 1355.3977, -1692.1660,
        -16.854889
    ],
    '8.00':
    [-1898.622, -1760.1617, 7816.228, 0.0, 4101.7017, -2539.0099, -27.502527],
    '10.0': [
        -2929.277, -2756.6884, 20483.217, 0.0, 11095.8257, -3328.0227,
        -39.169742
    ],
    '12.0': [
        -4182.363, -3975.6195, 50939.941, 0.0, 28724.6969, -3999.5813,
        -52.024708
    ],
    '15.0': [
        -6479.576, -6221.5028, 195570.801, 0.0, 118778.9520, -4694.7019,
        -71.996883
    ]
}

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

states_print = 3  # How many energies do you want?

# Parallel of this version not implemented
parallel = False  # Do you want trallel version?
Nprocessors = 6  # Number of processors

Sysem = "PJM"

############################
### Potential definition ###
############################
if Sysem == "Pionless":
    # Physical system and benchmark
    NState = 20  #Number of basys states
    order = 500  # Integration order
    m = 938.858
    mu = m / 2.
    hbar = 197.327
    mh2 = hbar**2 / (2 * mu)
    Rmax = 10  #Max R integration
    omegas = [1.0]
    L = 0
    potargs = [MuOmegaSquare]

    def pot_local(r, argv):
        return -505.1703491 * np.exp(-4. * r**2)

    interaction = "Local"

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

elif Sysem == "Hiyama_lambda_alpha":  # E = -3.12 MeV
    # Hyama non local benchmark
    NState = 20  #Number of basys states
    Rmax = 30
    order = 500
    m_alpha = 3727.379378
    m_lambda = 1115.683
    mu = (m_alpha * m_lambda) / (m_alpha + m_lambda)
    hbar = 197.327
    mh2 = hbar**2 / (2 * mu)
    omegas = [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12,
        0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3
    ]
    potargs = []
    L = 0
    interaction = "NonLocal"

    def pot_nonlocal(rl, rr, argv):
        ii = z = complex(0, 1)
        v1, a1, b1, c1 = -0.3706, -0.1808 - 0.4013, -0.1808 - 0.4013, (
            -0.1808 + 0.4013) * 2
        v2, a2, b2, c2 = -12.94, -0.1808 - 0.9633, -0.1808 - 0.9633, (
            -0.1808 + 0.9633) * 2
        v3, a3, b3, c3 = -331.2, -0.1808 - 2.930, -0.1808 - 2.930, (
            -0.1808 + 2.930) * 2
        Vvv = (ii**L) * (v1 * spherical_jn(L, -ii * c1 * rr * rl) * np.exp(
            a1 * rr**2 + b1 * rl**2) + v2 * spherical_jn(
                L, -ii * c2 * rr * rl) * np.exp(a2 * rr**2 + b2 * rl**2) +
                         v3 * spherical_jn(L, -ii * c3 * rr * rl
                                           ) * np.exp(a3 * rr**2 + b3 * rl**2))
        # this function is high unstable for large r (it gives NaN but it should give 0.)
        return np.nan_to_num(Vvv.real)

    def pot_local(r, argv):
        Vvv = (-17.49) * np.exp(-0.2752 * (r**2)) + (
            -127.0) * np.exp(-0.4559 *
                             (r**2)) + (497.8) * np.exp(-0.6123 * (r**2))
        return Vvv

elif Sysem == "PJM":
    # Prague-Jerusalem-Manchester effective A-1 interaction
    NState = 20  #Number of basys states
    Rmax = 30
    order = 500
    omegas = [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12,
        0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.3
    ]
    L = 0

    Ncore = 4  # number of core particles (capital A in docu)
    coreosci = .16  # oscillator parameter for the frozen core wave function

    mpi = '137'
    m = 938.858
    mu = Ncore * m / (Ncore + 1.)
    hbar = 197.327
    mh2 = hbar**2 / (2 * mu)

    interaction = "Local"
    Lamb = '4.00'
    LeC = lec_list_c[mpi][Lamb][0]  # PiLess LO 2-body LEC
    LeD = lec_list_c[mpi][Lamb][2]  # PiLess LO 3-body LEC

    potargs = [coreosci, Ncore, float(Lamb), LeC, LeD]

    def pot_nonlocal(rl, rr, argv):
        ii = z = complex(0, 1)
        v1, a1, b1, c1 = -0.3706, -0.1808 - 0.4013, -0.1808 - 0.4013, (
            -0.1808 + 0.4013) * 2
        v2, a2, b2, c2 = -12.94, -0.1808 - 0.9633, -0.1808 - 0.9633, (
            -0.1808 + 0.9633) * 2
        v3, a3, b3, c3 = -331.2, -0.1808 - 2.930, -0.1808 - 2.930, (
            -0.1808 + 2.930) * 2
        Vvv = (ii**L) * (v1 * spherical_jn(L, -ii * c1 * rr * rl) * np.exp(
            a1 * rr**2 + b1 * rl**2) + v2 * spherical_jn(
                L, -ii * c2 * rr * rl) * np.exp(a2 * rr**2 + b2 * rl**2) +
                         v3 * spherical_jn(L, -ii * c3 * rr * rl
                                           ) * np.exp(a3 * rr**2 + b3 * rl**2))
        # this function is high unstable for large r (it gives NaN but it should give 0.)
        return np.nan_to_num(Vvv.real)

    def pot_local(r, argv):

        Vvv = argv[3] * (8 * (-1 + argv[1])) / (4 + (
            (-1 + argv[1]) * argv[2]**2) / (argv[0] * argv[1]))**1.5 * np.exp(
                -(argv[0] * argv[1] * argv[2]**2) /
                (4 * argv[0] * argv[1] + (-1 + argv[1]) * argv[2]**2) * r**2)
        +argv[4] / 2. * 64 * argv[0]**3 * (-2 + argv[1]) * (-1 + argv[1]) * (
            argv[1] /
            (16 * argv[0]**2 * argv[1] + 4 * argv[0] *
             (-1 + 3 * argv[1]) * argv[2]**2 +
             (-2 + argv[1]) * argv[2]**4))**1.5 * np.exp(
                 -(2 * argv[0] * argv[1] * argv[2]**2 *
                   (2 * argv[0] + argv[2]**2)) /
                 (16 * argv[0]**2 * argv[1] + 4 * argv[0] *
                  (-1 + 3 * argv[1]) * argv[2]**2 +
                  (-2 + argv[1]) * argv[2]**4) * r**2) + argv[4] / 2. * (64 * (
                      -2 + argv[1]) * (-1 + argv[1])) / (
                          ((4 * argv[0] + argv[2]**2) *
                           (4 * argv[0] * argv[1] +
                            (-2 + argv[1]) * argv[2]**2)) /
                          (argv[0]**2 * argv[1]))**1.5 * np.exp(
                              -(2 * argv[0] * argv[1] * argv[2]**2) /
                              (4 * argv[0] * argv[1] +
                               (-2 + argv[1]) * argv[2]**2) * r**2)
        return Vvv

else:
    print("ERROR: I do not know the system you want")
    quit()

if __name__ == '__main__':
    print("parallel: ", parallel)
    if parallel:
        p = Pool(Nprocessors)
        print("Numbero fo CPU: ", cpu_count())

    print("")
    print("--- " + Sysem + "---")
    print("# of states : " + str(NState))
    print("Max R       : " + str(Rmax))
    print("Gauss order : " + str(order))
    print("Mass        : " + str(mu))
    print("hbar        : " + str(hbar))
    print("h^2/2m      : " + str(np.round(mh2, 3)))

    x, w = np.polynomial.legendre.leggauss(order)
    # Translate x values from the interval [-1, 1] to [a, b]
    a = 0.0
    b = Rmax
    t = 0.5 * (x + 1) * (b - a) + a
    gauss_scale = 0.5 * (b - a)

    val_omega = []
    ene_omega = []
    for omega in omegas:

        ###########################
        ### All vectors to zero ###
        ###########################
        H = np.zeros((NState, NState))
        K = np.zeros((NState, NState))
        V = np.zeros((NState, NState))
        Vl = np.zeros((NState, NState))
        U = np.zeros((NState, NState))
        nu = mu * omega / (2 * hbar)
        print("Omega       : " + str(omega))
        print("nu          : " + str(np.round(nu, 3)))
        print(" ")

        print("Creation of integration array: ")
        start_time = timeit.default_timer()
        start_time2 = start_time

        psiRN = np.zeros((order, NState))
        ddpsiRN = np.zeros([order, NState])
        VlocRN = np.zeros(order)
        for y in range(NState):
            for x in range(order):
                psiRN[x, y] = psi(t[x], y, L, nu)
                ddpsiRN[x, y] = ddpsi(t[x], y, L, nu)

        print(" >> Wave function: (",
              timeit.default_timer() - start_time, " s )")
        start_time = timeit.default_timer()
        VlocRN[:] = pot_local(t[:], potargs)
        print(" >> Local potential:  (",
              timeit.default_timer() - start_time, " s )")
        start_time = timeit.default_timer()

        if (interaction == "NonLocal"):
            VnolRN = np.fromfunction(
                lambda x, y: pot_nonlocal(t[x], t[y], potargs), (order, order),
                dtype=int)
            print(" >> NonLocal potential:  (",
                  timeit.default_timer() - start_time, " s )")
            start_time = timeit.default_timer()

        #psiRN   = np.fromfunction(lambda x, y:  psi(t[x], y, L, nu)  , (order,NState), dtype=int)
        #ddpsiRN = np.fromfunction(lambda x, y:  ddpsi(t[x], y, L, nu)  , (order,NState), dtype=int)
        #VlocRN  = np.fromfunction(lambda x:  pot_local(t[x],[]) , order, dtype=int)

        print("Array creation time:",
              timeit.default_timer() - start_time2, " s")
        start_time = timeit.default_timer()

        print(" ")
        print("Array integration:")

        for i in np.arange(NState):
            for j in np.arange(i + 1):
                U[i][j] = np.sum(
                    psiRN[:, i] * psiRN[:, j] * w[:]) * gauss_scale
                K[i][j] = np.sum(
                    psiRN[:, i] * ddpsiRN[:, j] * w[:]) * gauss_scale
                Vl[i][j] = np.sum(
                    psiRN[:, i] * VlocRN[:] * psiRN[:, j] * w[:]) * gauss_scale
                if (interaction == "NonLocal"):
                    for k in range(order):
                        V[i][j] = V[i][j] + 4. * np.pi * np.sum(
                            t[:] * VnolRN[k, :] * psiRN[:, j] * w[:]
                        ) * psiRN[k, i] * t[k] * w[k] * gauss_scale**2

        print("Integration time:", timeit.default_timer() - start_time, " s")
        start_time = timeit.default_timer()

        K = -mh2 * K
        H = V + Vl + K
        V = V
        Vl = Vl

        for i in np.arange(NState):
            for j in np.arange(0, i):
                H[j][i] = H[i][j]
                V[j][i] = V[i][j]
                Vl[j][i] = Vl[i][j]
                K[j][i] = K[i][j]
                U[j][i] = U[i][j]
        # Check unitarity:
        if np.sum(abs(np.eye(NState) - U)) > 0.1 * NState**2:
            print(" ")
            print("WARNING: omega = ", omega)
            print("   >>  unitarity condition not satisfied: ")
            print("   >>  average difference with unity matrix:",
                  np.round(np.sum(abs(np.eye(NState) - U)) / NState**2, 2))
            #print(np.round(U,2))
            print("--------------------------")
            print(" ")
            print(" ")
            print(" ")
            continue

        debug = False
        if debug:
            print("Gauss scale: ", gauss_scale)
            print(" ")
            print("U: ")
            print(np.round(U, 2))
            print("Vl: ")
            print(np.around(Vl, 2))
            print("V:  ")
            print(np.around(V, 2))
            print("K: ")
            print(np.around(K))
            print("H: ")
            print(np.around(H))

            mat = np.matrix(U)
            with open('Unity.txt', 'w') as f:
                for line in mat:
                    np.savetxt(f, line, fmt='%.2f')
            mat = np.matrix(V)
            with open('Poten.txt', 'w') as f:
                for line in mat:
                    np.savetxt(f, line, fmt='%.2f')
            mat = np.matrix(K)
            with open('Kinet.txt', 'w') as f:
                for line in mat:
                    np.savetxt(f, line, fmt='%.2f')
            mat = np.matrix(H)
            with open('Hamil.txt', 'w') as f:
                for line in mat:
                    np.savetxt(f, line, fmt='%.2f')

        # Diagonalize
        print(" ")
        print("Diagonalization:")
        for i in np.arange(NState, NState + 1):
            #for i in np.arange(NState+1):
            val, vec = np.linalg.eig(H[:i, :i])
            z = np.argsort(val)
            z = z[0:states_print]
            energies = (val[z])
            print("states: " + str(i) + "  Energies: " + str(energies))
        print("Diagonalization time:",
              timeit.default_timer() - start_time, " s")
        print("--------------------------")
        print(" ")
        print(" ")
        print(" ")
        ene_omega.append(energies[0])
        val_omega.append(omega)
    plt.plot(val_omega, ene_omega, 'ko', lw=2, label="{} ".format(i))
    plt.show()

    if parallel:
        print("Parallel closing")
        p.close()
    quit()

    # plotting wave function
    xx = np.linspace(0, step * NState, NState)
    plt.figure(figsize=(10, 8))
    for i in range(len(z)):
        y = []
        y = np.append(y, vec[:, z[i]])
        #y = np.append(y,0)
        #y = np.insert(y,0,0)
        plt.plot(xx, y, 'k--', lw=2, label="{} ".format(i))
        #plt.plot(xx,Vc,color='r',alpha=0.2)
        plt.xlabel('r', size=14)
        plt.ylabel('$\psi$(r)', size=14)
    #plt.ylim(-1,1)
    plt.legend()
    plt.title(
        'normalized wavefunctions for a harmonic oscillator using finite difference method',
        size=14)
    #plt.show()

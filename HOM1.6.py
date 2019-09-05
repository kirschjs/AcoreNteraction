import numpy as np
from scipy.special import eval_genlaguerre, iv
from scipy.integrate import quad, dblquad, fixed_quad, quadrature

from scipy import integrate
import matplotlib.pyplot as plt
import sys
import datetime
from multiprocessing import Lock, Process, Queue, current_process, Pool, cpu_count
from functools import partial

H = np.zeros((1, 1))
K = np.zeros((1, 1))
V = np.zeros((1, 1))
Vl = np.zeros((1, 1))
U = np.zeros((1, 1))

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


#######################################
######## Integrating operators ########
#######################################
def kin(r):
    global ARGS
    n, l, n1, l1, nu = ARGS
    if l == l1:
        return psi(r, n, l, nu) * ddpsi(r, n1, l1, nu)
    return 0


def psi2(r):
    n, l, n1, l1, nu = ARGS
    if l == l1:
        return psi(r, n, l, nu) * psi(r, n1, l1, nu)
    return 0


def pot_loc(r):
    global ARGS
    n, l, n1, l1, nu, MuOmegaSquare, mh2 = ARGS
    if l == l1:
        return psi(r, n, l, nu) * (pot_local(r, MuOmegaSquare) + mh2 * l *
                                   (l + 1) / r**2) * psi(r, n1, l1, nu)
    return 0

    #def pot_nol_q(erre): #rl,rr):
    #global ARGS
    #rl = erre[0]
    #rr = erre[1]
    #n, l, n1, l1, nu, MuOmegaSquare, mh2 = ARGS
    #if l == l1:
    #    Vvv = pot_nonlocal(rr, rl, l)
    #    return psi(rl, n, l, nu) * Vvv * psi(rr, n1, l1, nu)


#return 0


def pot_nol(rl, rr):
    global ARGS
    n, l, n1, l1, nu, MuOmegaSquare, mh2 = ARGS
    if l == l1:
        Vvv = pot_nonlocal(rr, rl, l)
        return (4. * np.pi) * (
            rr * rl) * psi(rl, n, l, nu) * Vvv * psi(rr, n1, l1, nu)
    return 0


def int_K(i, more, j):
    global V, Vl, K, ARGS
    L, nu, mu, omega, mh2, Rmax, order, t, w = more

    ARGS = i, L, j, L, nu
    K[i][j] = fixed_quad(kin, 0.0, Rmax, n=order)[0]
    if (K[i][j] != K[i][j]):
        K[i][j] = quad(kin, 0.0, Rmax)[0]
        if (K[i][j] != K[i][j]):
            print("kinetic", i, j)
            quit()
    return K[i][j]


def int_U(i, more, j):
    global V, Vl, K, ARGS
    L, nu, mu, omega, mh2, Rmax, order, scheme, t, w = more
    U[i][j] = quad(psi2, 0.0, np.inf, args=(i, L, j, L, nu))[0]
    return U[i][j]


def int_Vl(i, more, j):
    global V, Vl, K, ARGS
    L, nu, mu, omega, mh2, Rmax, order, t, w = more
    ARGS = i, L, j, L, nu, mu * omega**2, mh2

    #Vl[i][j] = quad(pot_loc, 0.0, Rmax)[0]
    Vl[i][j] = sum(w * pot_loc(t))
    #Vl[i][j] = fixed_quad(pot_loc, 0.0, Rmax, n=order)[0]
    #if (Vl[i][j] != Vl[i][j]):
    #    Vl[i][j] = quad(pot_loc, 0.0, Rmax)[0]
    #    if (Vl[i][j] != Vl[i][j]):
    #        print("potential", i, j)
    #        quit()
    return Vl[i][j]


def intermediate_quad(xx):

    global t, w
    #return quad(pot_nol, 0, Rmax, args=(x, ))[0]
    #return quadrature(pot_nol,0.0, Rmax, args=(x,), tol=1.49e-04, rtol=1.49e-04, maxiter=order, vec_func=False, miniter=order)[0]
    #return fixed_quad(pot_nol, 0, Rmax, args=(x, ),n=200)[0]
    partial = np.zeros(order)
    for i in np.arange(order):
        partial = partial + w[i] * pot_nol(xx, t[i])
    return partial  #* 0.5*(b - a)


def int_V(i, more, j):
    global V, Vl, K, ARGS
    global t, w
    L, nu, mu, omega, mh2, Rmax, order, t, w = more
    ARGS = i, L, j, L, nu, mu * omega**2, mh2
    V[i][j] = sum(w * intermediate_quad(t))

    #* 0.5*(b - a)
    #quadrature(lambda x:intermediate_quad(x), 0.0, Rmax, args=(), tol=1.49e-03, rtol=1.49e-03, maxiter=order, vec_func=False, miniter=order)[0]

    #V[i][j] = quad(lambda x:intermediate_quad(x), 0.0, Rmax)[0]
    return V[i][j]


########################################
### Harmonic Hoscillator bases input ###
########################################

NState = 6  # Number of basys states
Rmax = 50  # Max R integration
order = 154  # Integration order
states_print = 3  # How many energies do you want?
parallel = True  # Do you want trallel version?
Nprocessors = 3  # Number of processors
Sysem = "Hiyama_lambda_alpha"

two_dimension_quadrature = True

############################
### Potential definition ###
############################
if Sysem == "Pionless":
    # Physical system and benchmark
    NState = 50  #Number of basys states
    m = 938.858
    mu = m / 2.
    hbar = 197.327
    mh2 = hbar**2 / (2 * mu)
    Rmax = 10  #Max R integration
    omegas = [1.]
    L = 0

    def pot_local(r, MuOmegaSquare):
        return -505.1703491 * np.exp(-4. * r**2)

    interaction = "Local"

elif Sysem == "HObenchmark":
    # HO  benchmark
    m = 1
    mu = m
    hbar = 1
    omegas = [1.]
    L = 1

    def pot_local(r, MuOmegaSquare):
        return 0.5 * MuOmegaSquare * r**2

    interaction = "Local"

elif Sysem == "Hiyama_lambda_alpha":  # E = -3.12 MeV
    # Hyama non local benchmark
    NState = 5  # Number of basys states
    Rmax = 2
    m_alpha = 3727.379378
    m_lambda = 1115.683
    mu = (m_alpha * m_lambda) / (m_alpha + m_lambda)
    hbar = 197.327
    mh2 = hbar**2 / (2 * mu)
    omegas = [1.]
    L = 0
    interaction = "NonLocal"

    #interaction="Local"


    def pot_nonlocal(rl, rr, arguments):
        #V = -200. * np.exp(-0.2 * (rr+rl)**2 )
        Vvv = (
            (-0.3706) * np.exp(-0.1808 * (rr + rl)**2 - 0.4013 * (rr - rl)**2)
            +
            (-12.94) * np.exp(-0.1808 * (rr + rl)**2 - 0.9633 * (rr - rl)**2) +
            (-331.2) * np.exp(-0.1808 * (rr + rl)**2 - 2.930 * (rr - rl)**2))
        return Vvv

    def pot_local(r, arguments):
        Vvv2 = ((-17.49) * np.exp(-0.2752 * (r**2)) +
                (-127.0) * np.exp(-0.4559 *
                                  (r**2)) + (497.8) * np.exp(-0.6123 * (r**2)))
        return Vvv2

elif Sysem == "P_wave_cluster_Johannes":
    # non local benchmark
    NState = 200  #Number of basys states
    Rmax = 2  #Radius after which the interaction is practically zero
    m_cluster = 3727.379378  #Mass first cluster
    m_fermion = 938.858
    mu = (m_alpha * m_lambda) / (m_alpha + m_lambda)
    hbar = 197.327
    mh2 = hbar**2 / (2 * mu)
    omegas = [0.5]
    L = 1
    interaction = "NonLocal"

    def pot_nonlocal(rl, rr, MuOmegaSquare):
        Vvv = (
            -0.3706) * np.exp(-0.1808 * (rr + rl)**2 - 0.4013 * (rr - rl)**2)
        return Vvv

    def pot_local(r, MuOmegaSquare):
        Vvv = (-17.49) * np.exp(-0.2752 * r**2)
        return Vvv

else:
    print("ERROR: I do not know the system you want")
    quit()

###########################
### All vectors to zero ###
###########################
H = np.zeros((NState, NState))
K = np.zeros((NState, NState))
V = np.zeros((NState, NState))
Vl = np.zeros((NState, NState))
U = np.zeros((NState, NState))

if __name__ == '__main__':

    print("parallel: ", parallel)
    if parallel:
        p = Pool(Nprocessors)
        print("Numbero fo CPU: ", cpu_count())

    #mh2 = hbar ** 2 / (2. * mu)
    print(datetime.datetime.now())
    print("")
    print("--- " + Sysem + "---")
    print("# of states : " + str(NState))
    print("Max R       : " + str(Rmax))
    print("Gauss order : " + str(order))
    print("Mass        : " + str(mu))
    print("hbar        : " + str(hbar))
    print("h^2/2m      : " + str(np.round(mh2, 3)))

    debug = False
    if debug:

        omega = 1
        nu = mu * omega / (2 * hbar)

        print(" ")
        print(
            "(Line 290) DEBUG: Most complicated integral check (Omega = 1. ) :"
        )

        def testintegral(x, y):
            #return((x+y+1)**-1)
            #return(np.exp(-(x+y)**2))
            return psi(x, 2, 1, 0.5) * np.exp(-16. *
                                              (x + y)**2) * psi(y, 2, 1, 0.5)

        def first_int_fix(rr):
            #order = 10
            return fixed_quad(testintegral, 0.0, Rmax, args=(rr, ), n=order)[0]

        def first_int(rr):
            return quad(testintegral, 0.0, Rmax, args=(rr, ))[0]

        def first_int_quad(rr):
            return quadrature(
                testintegral,
                0.0,
                Rmax,
                args=(rr, ),
                tol=1.49e-04,
                rtol=1.49e-04,
                maxiter=order,
                vec_func=False,
                miniter=order)[0]

        order = 20
        Rmax = 2

        x, w = np.polynomial.legendre.leggauss(order)
        # Translate x values from the interval [-1, 1] to [a, b]

        a = 0.0
        b = Rmax
        t = 0.5 * (x + 1) * (b - a) + a

        #f = lambda x: sum(w * np.exp((t+x)))* 0.5*(b - a)


        def f(xx):
            partial = np.zeros(order)
            for i in np.arange(order):
                partial = partial + w[i] * testintegral(xx, t[i])
            return partial * 0.5 * (b - a)
            #return np.sum( w[:] * testintegral(xx[:],t[:])) * 0.5*(b - a)

        #f = lambda xx:  np.sum( w[:] * testintegral(xx,t[:])) * 0.5*(b - a)
        #sum(w * psi(x, 10, 1, 0.5)*np.exp(-16.*(x+t)**2)*psi(t, 10, 1, 0.5))

        integ = sum(w * f(t)) * 0.5 * (b - a)
        print("polinomial hand : ", integ)

        integ = quad(lambda x: first_int(x), 0.0, Rmax)[0]
        print("double 1d int:    ", integ)

        integ = quad(lambda x: first_int_fix(x), 0.0, Rmax)[0]
        print("mix double 1d int:", integ)

        integ = fixed_quad(lambda x: first_int_fix(x), 0.0, Rmax, n=order)[0]
        print("fix double 1d int:", integ)

        #integ = quadrature(lambda x:first_int_quad(x), 0.0, Rmax, args=(), tol=1.49e-03, rtol=1.49e-03, maxiter=order, vec_func=False, miniter=order)[0]
        #print("quadrature double 1d int:",integ      )

        ARGS = 5, L, 5, L, nu, mu * omega**2, mh2
        print("Complete to Rmax: ",
              dblquad(
                  testintegral,
                  0.0,
                  Rmax,
                  lambda x: 0.,
                  lambda x: Rmax,
                  epsabs=1.e-03,
                  epsrel=1.e-03)[0])
        print("Complete to inf:  ",
              dblquad(
                  testintegral,
                  0.0,
                  np.inf,
                  lambda x: 0.,
                  lambda x: np.inf,
                  epsabs=1.e-03,
                  epsrel=1.e-03)[0])

        exit()

    x, w = np.polynomial.legendre.leggauss(order)
    # Translate x values from the interval [-1, 1] to [a, b]
    a = 0.0
    b = Rmax
    t = 0.5 * (x + 1) * (b - a) + a
    gauss_scale = 0.5 * (b - a)

    for omega in omegas:
        nu = mu * omega / (2 * hbar)
        print("Omega       : " + str(omega))
        print("nu          : " + str(np.round(nu, 3)))
        print(" ")
        print("Matrix creation:")

        more = L, nu, mu, omega, mh2, Rmax, order, t, w

        for i in np.arange(NState):
            a_string = str(np.round(100. * i / ((NState + 1)), 1)) + " %"
            sys.stdout.write(a_string + " " * (78 - len(a_string)) + "\r")

            func = partial(int_Vl, i, more)
            if (parallel):
                Vl[i][:i + 1] = (p.map(func, np.arange(i + 1)))
            else:
                for j in np.arange(i + 1):
                    func(j)

            if (interaction == "NonLocal"):
                func = partial(int_V, i, more)
                if (parallel):
                    V[i][:i + 1] = (p.map(func, np.arange(i + 1)))
                else:
                    for j in np.arange(i + 1):
                        V[i][j] = func(j)

            func = partial(int_K, i, more)
            if (parallel):
                K[i][:i + 1] = (p.map(func, np.arange(i + 1)))
            else:
                for j in np.arange(i + 1):
                    func(j)

        K = -mh2 * K
        V = V * gauss_scale**2
        Vl = Vl * gauss_scale
        H = V + Vl + K

        for i in np.arange(NState):
            for j in np.arange(0, i):
                H[j][i] = H[i][j]
                V[j][i] = V[i][j]
                Vl[j][i] = Vl[i][j]
                K[j][i] = K[i][j]

        print(mh2)

        debug = True
        if debug:
            print("Gauss scale: ", gauss_scale)
            print("Vl: ")
            print(np.around(Vl, 2))
            print("V:  ")
            print(np.around(V, 2))
            print("K: ")
            print(np.around(K))
            print("H: ")
            print(np.around(H))

        print("DONE")
        print(datetime.datetime.now())
        print("")

        debug = False
        if debug:
            #mat = np.matrix(U)
            #with open('Unity.txt', 'w') as f:
            #    for line in mat:
            #print(line)
            #print(" ")
            #        np.savetxt(f, line, fmt='%.2f')
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
        for i in np.arange(NState):
            val, vec = np.linalg.eig(H[:i, :i])
            z = np.argsort(val)
            z = z[0:states_print]
            energies = (val[z])
            print("states: " + str(i) + "  Energies: " + str(energies))
        print("--------------------------")
        print(" ")
        print(" ")
        print(" ")

    if parallel:
        print("Parallel closing")
        p.close()
    quit()

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

import numpy as np
from scipy.special import eval_genlaguerre, iv
from scipy.integrate import quad,dblquad,fixed_quad
import matplotlib.pyplot as plt
import sys
import datetime
import quadpy
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
    N = np.sqrt(np.sqrt(2. * (nu ** 3) / np.pi) * (
                2. ** (n + 2 * l + 3)  * nu ** l * np.exp(log_fact(n) - log_d_fact(2 * n + 2 * l + 1) ) ))
    psi = r * N * r ** l * np.exp(-nu * r ** 2) * eval_genlaguerre(n, l + 0.5, 2 * nu * r ** 2)
    return psi


def ddpsi(r, n, l, nu):
    N = np.sqrt(np.sqrt(2. * (nu ** 3) / np.pi) * (
            2. ** (n + 2 * l + 3) * nu ** l * np.exp(log_fact(n) - log_d_fact(2 * n + 2 * l + 1))))
    ddpsi = np.exp(-r ** 2 * nu) * (16. * r ** (4 + l - 1) * nu ** 2 * eval_genlaguerre(n - 2, l + 2.5, 2 * nu * r ** 2)
                                    + 4. * r ** (2 + l - 1) * nu * (-3 - 2 * l + 4 * r ** 2 * nu) * eval_genlaguerre(
                n - 1, l + 1.5, 2 * nu * r ** 2)
                                    + r ** (l - 1) * (l * (l + 1) - 2 * (
                        3 + 2 * l) * r ** 2 * nu + 4 * r ** 4 * nu ** 2) * eval_genlaguerre(n, l + 0.5,
                                                                                            2 * nu * r ** 2))
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
        return psi(r, n, l, nu) * ( pot_local(r, MuOmegaSquare) + mh2 * l*(l+1) / r**2) * psi(r, n1, l1, nu)
    return 0

def pot_nol_q(erre): #rl,rr):
    global ARGS
    rl = erre[0]
    rr = erre[1]
    n, l, n1, l1, nu, MuOmegaSquare, mh2 = ARGS
    if l == l1:
        Vvv = pot_nonlocal(rr, rl, l)
        return psi(rl, n, l, nu) * Vvv * psi(rr, n1, l1, nu)
    return 0

def pot_nol(rl,rr):
    global ARGS
    n, l, n1, l1, nu, MuOmegaSquare, mh2 = ARGS
    if l == l1:
        Vvv = pot_nonlocal(rr, rl, l)
        return psi(rl, n, l, nu) * Vvv * psi(rr, n1, l1, nu)
    return 0

def int_K(i,more,j):
        global V,Vl,K,ARGS
        L,nu,mu,omega,mh2,Rmax,order,scheme = more


        ARGS = i, L, j, L, nu
        K[i][j] = fixed_quad(kin, 0.0, Rmax, n=order)[0]
        if (K[i][j] != K[i][j]):
            K[i][j] = quad(kin, 0.0, Rmax)[0]
            if (K[i][j] != K[i][j]):
                print("kinetic", i, j)
                quit()
        return K[i][j]


def int_U(i, more, j):
        global V,Vl,K,ARGS
        L,nu,mu,omega,mh2,Rmax,order,scheme = more
        U[i][j] = quad(psi2, 0.0, np.inf, args=(i,L,j,L,nu))[0]
        return U[i][j]

def int_Vl(i, more, j):
        global V,Vl,K,ARGS
        L,nu,mu,omega,mh2,Rmax,order,scheme = more

        ARGS = i, L, j, L, nu, mu * omega ** 2, mh2
        Vl[i][j] = fixed_quad(pot_loc, 0.0, Rmax, n=order)[0]
        if (Vl[i][j] != Vl[i][j]):
            Vl[i][j] = quad(pot_loc, 0.0, Rmax)[0]
            if (Vl[i][j] != Vl[i][j]):
                print("potential", i, j)
                quit()
        return Vl[i][j]


def int_V(i, more, j):
        global V,Vl,K,ARGS
        L,nu,mu,omega,mh2,Rmax,order,scheme = more
        Rmax=2.
        #scheme = quadpy.quadrilateral.witherden_vincent_21()
        scheme = quadpy.quadrilateral.sommariva_54()
        ARGS = i, L, j, L, nu, mu * omega ** 2, mh2
        if (interaction == "NonLocal"):
            if two_dimension_quadrature:
                V[i][j] = scheme.integrate(pot_nol_q, [[[0.0, 0.0], [Rmax, 0.0]], [[0.0, Rmax], [Rmax, Rmax]]])
            else:
                V[i][j] = dblquad(pot_nol, 0.0, Rmax, lambda x: 0.,
                lambda x: Rmax,epsabs=1.49e-03, epsrel=1.49e-03)[0]
        return V[i][j]
#        return (v,vl,k)







########################################
### Harmonic Hoscillator bases input ###
########################################

NState   = 200       #Number of basys states
Rmax     = 50        #Max R integration
order    = 200       #Integration order
states_print = 3     #How many energies do you want?
parallel = True     # Do you want trallel version?
Nprocessors = 4      # Number of processors
Sysem = "Hiyama_lambda_alpha"

two_dimension_quadrature = True






############################
### Potential definition ###
############################
if Sysem == "Pionless" :
# Physical system and benchmark
    NState = 150     #Number of basys states
    m     = 938.858
    mu    = m/2.
    hbar  = 197.327
    mh2   = hbar**2/(2*m)
    Rmax   = 50     #Max R integration
    omegas  = [0.5]
    L=0
    def pot_local(r, MuOmegaSquare):
        return -505.1703491 * np.exp(- 4. * r ** 2)
    interaction="Local"

elif Sysem == "HObenchmark"  :
# HO  benchmark
    m      = 1
    mu     = m
    hbar   = 1
    omegas  = [1.]
    L      = 1
    def pot_local(r, MuOmegaSquare):
        return 0.5*MuOmegaSquare*r**2
    interaction="Local"

elif Sysem == "Hiyama_lambda_alpha"  :
# Hyama non local benchmark
    NState = 200     #Number of basys states
    Rmax   = 2
    m_alpha      = 3727.379378
    m_lambda     = 1115.683
    mu      = (m_alpha*m_lambda)/(m_alpha+m_lambda)
    hbar    = 197.327
    mh2     = hbar**2/(2*mu)
    omegas  = [0.5]
    L       = 0
    interaction="NonLocal"

    scheme = quadpy.quadrilateral.sommariva_54()
    scheme = quadpy.quadrilateral.witherden_vincent_21()

    def pot_nonlocal(rl, rr, MuOmegaSquare):
        V = ((-0.3706) * np.exp(-0.1808 * (rr+rl)**2 - 0.4013 * (rr-rl)**2) +
             (-12.94 ) * np.exp(-0.1808 * (rr+rl)**2 - 0.9633 * (rr-rl)**2) +
             (-331.2 ) * np.exp(-0.1808 * (rr+rl)**2 - 2.930  * (rr-rl)**2))
        return V

    def pot_local(r, MuOmegaSquare):
        V = ((-17.49) * np.exp(-0.2752*r**2) +
             (-127.0) * np.exp(-0.4559*r**2) +
             (497.8)  * np.exp(-0.6123*r**2))
        return V

elif Sysem == "P_wave_cluster_Johannes"  :
# Hyama non local benchmark
    NState = 200     #Number of basys states
    Rmax   = 2       #Radius after which the interaction is practically zero
    m_cluster      = 3727.379378  #Mass first cluster
    m_fermion      = 938.858
    mu      = (m_alpha*m_lambda)/(m_alpha+m_lambda)
    hbar    = 197.327
    mh2     = hbar**2/(2*mu)
    omegas  = [0.5]
    L       = 1
    interaction="NonLocal"


    def pot_nonlocal(rl, rr, MuOmegaSquare):
        V = (-0.3706) * np.exp(-0.1808 * (rr+rl)**2 - 0.4013 * (rr-rl)**2)
        return V

    def pot_local(r, MuOmegaSquare):
        V = (-17.49) * np.exp(-0.2752*r**2)
        return V


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

    mh2 = hbar ** 2 / (2. * mu)
    print(datetime.datetime.now())
    print("")
    print("--- "+ Sysem + "---")
    print("# of states : " + str(NState))
    print("Max R       : " + str(Rmax))
    print("Gauss order : " + str(order))
    print("Mass        : " + str(mu))
    print("hbar        : " + str(hbar))
    print("h^2/2m      : " + str(np.round(mh2,3)))

    omega = 1
    nu = mu * omega / (2 * hbar)

    debug=True
    if True:
        print(" ")
        print("Most complicated integral check (Omega = 1. )")
        #ARGS = NState-1, L, NState-1, L, nu, mu * omega ** 2, mh2
        ARGS = 20, L, 20, L, nu, mu * omega ** 2, mh2
        val = scheme.integrate(pot_nol_q,[[[0.0, 0.0], [Rmax, 0.0]], [[0.0, Rmax], [Rmax, Rmax]]])
        print("Approximation:    ", val)
        print("Complete to Rmax: ", dblquad(pot_nol, 0.0, Rmax, lambda x: 0.,
                    lambda x: Rmax,epsabs=1.e-03, epsrel=1.e-03)[0])
        print("Complete to inf:  ", dblquad(pot_nol, 0.0, np.inf, lambda x: 0.,
                    lambda x: np.inf,epsabs=1.e-03, epsrel=1.e-03)[0])
        print(" ")


    for omega in omegas:
            nu = mu * omega / (2 * hbar)
            print("Omega       : " + str(omega))
            print("nu          : " + str(np.round(nu,3)))
            print(" ")
            print("Matrix creation:")


            #scheme = quadpy.quadrilateral.witherden_vincent_21()
            scheme = quadpy.quadrilateral.sommariva_54()
    
            if (parallel):
                # I dont know why but i can not pass integration scheme
                # in a function if multiprocessing.
                more = L, nu, mu, omega, mh2, Rmax, order, 12
            else:
                more = L, nu, mu, omega, mh2, Rmax, order, scheme
            
            for i in np.arange(NState):
                    a_string = str(np.round(100. * i / ((NState + 1) ), 1)) + " %"
                    sys.stdout.write(a_string + " " * (78 - len(a_string)) + "\r")

                    func = partial(int_Vl, i, more)
                    if (parallel):
                        Vl[i][:i+1] = (p.map(func, np.arange(i+1)))
                    else:
                        for j in np.arange(i+1):
                            func(j)


                    if (interaction=="NonLocal"):
                        func = partial(int_V, i, more)
                        if (parallel):
                            V[i][:i+1] = (p.map(func, np.arange(i+1)))
                        else:
                            for j in np.arange(i+1):
                                func(j)

                    func = partial(int_K, i, more)
                    if (parallel):
                        K[i][:i+1] = (p.map(func, np.arange(i+1)))
                    else:
                        for j in np.arange(i+1):
                            func(j)


            K = - mh2*K
            H =   V + Vl + K


            for i in np.arange(NState):
                for j in np.arange(0,i):
                    H[j][i]=H[i][j]


            if debug:
                print(np.around(Vl))
                print(np.around(V))
                print(np.around(K))
                print(np.around(H))



            print("DONE")
            print(datetime.datetime.now())
            print("")



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
            val,vec=np.linalg.eig(H)
            z = np.argsort(val)
            z = z[0:states_print]
            energies=(val[z])
            print("Energies:" + str(energies))
            print("--------------------------")
            print(" ")
            print(" ")
            print(" ")





    if parallel:
        print("Parallel closing")
        p.close()
    quit()

    xx = np.linspace(0, step*NState,NState)
    plt.figure(figsize=(10,8))
    for i in range(len(z)):
        y = []
        y = np.append(y,vec[:,z[i]])
        #y = np.append(y,0)
        #y = np.insert(y,0,0)
        plt.plot(xx,y,'k--',lw=2, label="{} ".format(i))
        #plt.plot(xx,Vc,color='r',alpha=0.2)
        plt.xlabel('r', size=14)
        plt.ylabel('$\psi$(r)',size=14)
    #plt.ylim(-1,1)
    plt.legend()
    plt.title('normalized wavefunctions for a harmonic oscillator using finite difference method',size=14)
    #plt.show()


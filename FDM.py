import numpy as np
from scipy.special import eval_genlaguerre, iv
from scipy.integrate import quad,dblquad,fixed_quad
import matplotlib.pyplot as plt
import sys
import datetime

#import quadpy



######## Wave function ########
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


######## Integrating operators ########
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

def pot_nol(rl,rr):
    global ARGS
    n, l, n1, l1, nu, MuOmegaSquare, mh2 = ARGS
    if l == l1:
        Vvv = pot_nonlocal(rr, rl, l)
        return psi(rl, n, l, nu) * Vvv * psi(rr, n1, l1, nu)
    return 0




# Harmonic Hoscillator bases
NState = 15000     #Number of basys states
Rmax   = 20     #Max R integration
order  = 200       #Integration order

states_print = 3 #How many energies do you want?

Sysem = "HObenchmark"






if Sysem == "Pionless" :
# Physical system and benchmark
    NState = 5000     #Number of basys states
    m     = 938.858
    mu    = m/2.
    hbar  = 197.327
    mh2   = hbar**2/(2*m)
    Rmax   = 5     #Max R integration
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
    L      = 0
    def pot_local(r, MuOmegaSquare):
        return 0.5*MuOmegaSquare*r**2
    interaction="Local"

elif Sysem == "Hyama_lambda_alpha"  :
# HO  benchmark
    m_alpha      = 3727.379378
    m_lambda     = 1115.683
    mu      = (m_alpha*m_lambda)/(m_alpha+m_lambda)
    hbar    = 197.327
    mh2     = hbar**2/(2*mu)
    omegas  = [1.]
    L       = 0
    interaction="NonLocal"

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


#    quadpy.quadrilateral.integrate(
#    lambda x: numpy.exp(x[0]),
#    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
#    quadpy.quadrilateral.Product(quadpy.line_segment.GaussLegendre(4))
#    )
    interaction="NonLocal"

else:
    print("ERROR: I do not know the system you want")
    quit()



step=Rmax/NState
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
print("step        : " + str(np.round(step,3)))


for omega in omegas:
        nu = mu * omega / (2 * hbar)
        print("Omega       : " + str(omega))
        print("nu          : " + str(np.round(nu,3)))
        print(" ")

        #print(pot_nonlocal(0.5,0.75,1))
        #print(psi(0.75, 1, 0, nu))
        #ARGS = 1, 0, 1, 0, nu, mu * omega ** 2, mh2
        #print( dblquad(pot_nol, 0.0, Rmax, lambda x: 0.,
        #    lambda x: Rmax,epsabs=1.49e-03, epsrel=1.49e-03)[0])
        #exit()

        H  = np.zeros((NState,NState))
        K  = np.zeros((NState,NState))
        V  = np.zeros((NState,NState))
        Vl = np.zeros((NState,NState))
        U  = np.zeros((NState,NState))

        k=0
        print("Matrix creation:")


        for i in np.arange(NState):
            Vl[i][i] = pot_local(i*step,  mu*omega**2)
            K[i][i] = -2
            if (i < NState-1):
                K[i][i+1] = 1
            if (i > 0):
                K[i][i-1] = 1
            #for j in np.arange(i+1):
            #    K[j][i] = K[i][j]

                #if  (interaction == "NonLocal"):
                #    V[i][j] = dblquad(pot_nol, 0.0, Rmax, lambda x: 0.,
                #   lambda x: Rmax,epsabs=1.49e-03, epsrel=1.49e-03)[0]
#
#
#                Vl[j][i] = Vl[i][j]
  #              V[j][i] = V[i][j]
#                print(V[i][j], Vl[i][j])
        print("DONE")
        print(datetime.datetime.now())
       # print("")
       # print(np.round(K,1))

        K = - mh2*K/(step**2)
        H =   V + Vl + K




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
        energies=(val[z])#val[z][0])
        print("Energies:" + str(energies))






#quit()

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
plt.show()


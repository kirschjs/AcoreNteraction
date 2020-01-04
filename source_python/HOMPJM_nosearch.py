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


###  Interaction ###


# *****************************************************
def exchange_kernel(rl, rr, argv):
    #return 0
    rr2 = rr**2
    rl2 = rl**2

    # this term goes to the RHS of Eq.(12)
    # the 4Pi stems from exp(rr')=... and is not the one from the wave functions
    # which is included below via t[] in Vnonloc[...]
    V1ex = (4 * np.pi * 1j**Lrel) * ((zz1 * np.nan_to_num(
        spherical_jn(Lrel, 1j * bb1 * rr * rl),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0) * np.nan_to_num(
            np.exp(-aa1 * rr2 - gg1 * rl2), nan=0.0, posinf=1.0, neginf=-1.0)))

    return np.nan_to_num(np.real(V1ex))


def pot_nonlocal(rl, rr, argv):
    #return 0
    rr2 = rr**2
    rl2 = rl**2

    V1 = -mh2 * 4. * np.pi * zz1 * np.nan_to_num(
        np.exp(-aa1 * rr2 - gg1 * rl2), nan=0.0, posinf=1.0, neginf=-1.0) * (
            (4. * aa1 * bb1 * rl * rr) *
            ((1j**
              (Lrel - 1) * np.nan_to_num(
                  spherical_jn(Lrel - 1, 1j * bb1 * rr * rl),
                  nan=0.0,
                  posinf=1.0,
                  neginf=-1.0) * wigm * (2 * Lrel - 3)) +
             (1j**(Lrel + 1) * np.nan_to_num(
                 spherical_jn(Lrel + 1, 1j * bb1 * rr * rl),
                 nan=0.0,
                 posinf=1.0,
                 neginf=-1.0) * wigp * (2 * Lrel - 1))) +
            (-4. * aa1**2 * rr2 + 2 * aa1 - bb1**2 * rl2 + Lrel *
             (Lrel + 1) / rr2) * 1j**Lrel * np.nan_to_num(
                 spherical_jn(Lrel, 1j * bb1 * rr * rl),
                 nan=0.0,
                 posinf=1.0,
                 neginf=-1.0))

    # this is simple (still missing (4 pi i^l ) RR')
    V234 = -(4 * np.pi * 1j**Lrel) * ((zz2 * np.nan_to_num(
        spherical_jn(Lrel, 1j * bb2 * rr * rl),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0) * np.nan_to_num(
            np.exp(-aa2 * rr2 - gg2 * rl2), nan=0.0, posinf=1.0, neginf=-1.0
        )) + (zz3 * np.nan_to_num(
            spherical_jn(Lrel, 1j * bb3 * rr * rl),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        ) * np.nan_to_num(
            np.exp(-aa3 * rr2 - gg3 * rl2), nan=0.0, posinf=1.0, neginf=-1.0
        )) + (zz4 * np.nan_to_num(
            spherical_jn(Lrel, 1j * bb4 * rr * rl),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        ) * np.nan_to_num(
            np.exp(-aa4 * rr2 - gg4 * rl2), nan=0.0, posinf=1.0, neginf=-1.0)))

    # this function is high unstable for large r (it gives NaN but it should give 0.)
    return np.nan_to_num(np.real(V1 + V234))


def pot_local(r, argv):
    r2 = r**2
    VnnDI = nn1 * np.exp(-kk1 * r2)
    VnnnDIarm = nn2 * np.exp(-kk2 * r2)
    VnnnDIstar = nn3 * np.exp(-kk3 * r2)
    return VnnDI + VnnnDIarm + VnnnDIstar


# *****************************************************

#######################
### general options ###
#######################

pedantic = 0

NState = 20
Rmax = 120
order = 600

Lrel = 1  # Momento angolare

# physical system
mpi = '137'
m = 938.12
hbar = 197.327

#aumenta rcore e aumenti Lc zet = 0 : local, zet = 1 : non-local
# Lc4 = 0.6
#name, lec_list, Aradius_modifyer_fac,Aradius_modifyer_exp, interaction ="13L", lec_list_one_three,  0.02, "Local"
name, lec_list, Aradius_modifyer_fac, Aradius_modifyer_exp, interaction, zet = "13", lec_list_one_three, 0.004, 0., "NonLocal", 1.

# Lc3 = 0.65
#name, lec_list, Aradius_modifyer_fac,Aradius_modifyer_exp, interaction ="14L", lec_list_one_four,  0.02, "Local"
#name, lec_list, Aradius_modifyer_fac,Aradius_modifyer_exp, interaction ="14", lec_list_one_four,  0.00044, "NonLocal"

# Lc15 = 0.9
#name, lec_list, Aradius_modifyer_fac,Aradius_modifyer_exp, interaction ="1115L", lec_list_one_onefive,  0.009,0., "Local"
#name, lec_list, Aradius_modifyer, interaction ="1115", lec_list_one_onefive,  0.000412, "Local+NonLocal"

# Lc0 = 0.7
#name, lec_list, Aradius_modifyer_fac,Aradius_modifyer_exp, interaction ="04L", lec_list_unitary_scatt,  1.432,0., "Local"
#name, lec_list, Aradius_modifyer_fac, Aradius_modifyer_exp, interaction = "04L", lec_list_unitary_scatt, 1.432, 0., "Local"
#name, lec_list, Aradius_modifyer_fac,Aradius_modifyer_exp, interaction ="04", lec_list_unitary_scatt,  0.000414, "NonLocal"

Amin = 20
Amax = 21

Lmin = 3.5
Lmax = 9.5
dL = 2.5

cores = [20]  #range(Amin, Amax)
Lrange = np.arange(Lmin, Lmax, dL)

omegas = np.arange(0.05, 0.14, 0.025)
Aradius_modifyer_fac = np.arange(0.001, 0.01, 0.002)

resu = []

for Ncore in cores:

    print("\n\n=========== A Core " + str(Ncore) + " ============")

    for Lamb in Lrange:

        la = ('%-4.2f' % Lamb)[:4]
        try:
            LeC = lec_list[la][0]
            LeD = lec_list[la][1]
        except:
            print(
                'LECs not fitted for this cutoff.\n Use (unreliable) LECmodel fit.'
            )
            exit()

        for rad_fac in Aradius_modifyer_fac:

            coreosci = rad_fac * Ncore**((1. + Aradius_modifyer_exp) / 3.)

            mu = Ncore * m / (Ncore + 1.)
            mh2 = hbar**2 / (2 * mu)
            potargs = [coreosci, Ncore, float(Lamb), LeC, LeD]
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
            w3j_m2 = (Wigner3j(1, Lrel - 1, Lrel, 0, 0, 0).doit())**2
            w3j_p2 = (Wigner3j(1, Lrel + 1, Lrel, 0, 0, 0).doit())**2
            wigm = 0 if (w3j_m2 == 0) else float(w3j_m2.evalf())
            wigp = 0 if (w3j_p2 == 0) else float(w3j_p2.evalf())

            if pedantic:
                print("Lambda  = %2.2f  A = %d\n" % (Lamb, Ncore))
                print("a core  = %4.4f\n" % coreosci)
                print("LEC 2b = %+6.4e   LEC 3b  = %+6.4e\n" % (LeC, LeD))
                print(
                    '            1              2              3              4'
                )
                print("alpha  = %+6.4e    %+6.4e    %+6.4e    %+6.4e" %
                      (aa1, aa2, aa3, aa4))
                print("beta   = %+6.4e    %+6.4e    %+6.4e    %+6.4e" %
                      (bb1, bb2, bb3, bb4))
                print("gamma  = %+6.4e    %+6.4e    %+6.4e    %+6.4e" %
                      (gg1, gg2, gg3, gg4))
                print("kappa  = %+6.4e    %+6.4e    %+6.4e" % (kk1, kk2, kk3))
                print("eta    = %+6.4e    %+6.4e    %+6.4e" % (nn1, nn2, nn3))
                print("zeta   = %+6.4e    %+6.4e    %+6.4e    %+6.4e\n" %
                      (zz1, zz2, zz3, zz4))

            x, w = np.polynomial.legendre.leggauss(order)
            # Translate x values from the interval [-1, 1] to [a, b]
            a = 0.0
            b = Rmax
            t = 0.5 * (x + 1) * (b - a) + a
            gauss_scale = 0.5 * (b - a)

            # for each set of interaction parameters A,Lambda,core_osci
            # the Hamiltonian is diagonalized for the chose range of basis oscillator widths
            for omega in omegas:

                # Initialize vectors to zero.
                H = np.zeros((NState, NState))
                Hloc = np.zeros((NState, NState))
                Kin = np.zeros((NState, NState))
                Vnonloc = np.zeros((NState, NState))
                Vloc = np.zeros((NState, NState))
                U = np.zeros((NState, NState))
                Uex = np.zeros((NState, NState))
                psiRN = np.zeros((order, NState))
                ddpsiRN = np.zeros([order, NState])
                VlocRN = np.zeros(order)
                VexRN = np.zeros((order, order))

                nu = mu * omega / (2 * hbar)

                for y in range(NState):
                    for x in range(order):
                        psiRN[x, y] = psi(t[x], y, Lrel, nu)
                        ddpsiRN[x, y] = ddpsi(t[x], y, Lrel, nu)

                VlocRN[:] = pot_local(
                    t[:], potargs) + mh2 * Lrel * (Lrel + 1) / t[:]**2

                if (interaction == "NonLocal"):
                    VexRN = np.fromfunction(
                        lambda x, y: exchange_kernel(t[x], t[y], potargs),
                        (order, order),
                        dtype=int)
                    VnolRN = np.fromfunction(
                        lambda x, y: pot_nonlocal(t[x], t[y], potargs),
                        (order, order),
                        dtype=int)

                # fill the lower triangular matrices only and exploit the symmetry
                # ECCE: I reduced the calculation to one loop
                for i in np.arange(NState):
                    for j in np.arange(i + 1):
                        U[i][j] = np.sum(
                            psiRN[:, i] * psiRN[:, j] * w[:]) * gauss_scale
                        U[j][i] = U[i][j]
                        Kin[i][j] = np.sum(
                            psiRN[:, i] * ddpsiRN[:, j] * w[:]) * gauss_scale
                        Kin[j][i] = Kin[i][j]
                        Vloc[i][j] = np.sum(psiRN[:, i] * VlocRN[:] *
                                            psiRN[:, j] * w[:]) * gauss_scale
                        Vloc[j][i] = Vloc[i][j]
                        if (interaction == "NonLocal"):
                            for k in range(order):
                                Uex[i][j] = Uex[i][j] + 4. * np.pi * np.sum(
                                    t[:] * VexRN[k, :] * psiRN[:, j] * w[:]
                                ) * psiRN[k, i] * t[k] * w[k] * gauss_scale**2
                                #for k in range(order):
                                #Vnonloc[i][j] = Vnonloc[i][j] +
                                #np.sum(VnolRN[k, :] * psiRN[:, j] * w[:]) * psiRN[k, i] * w[k] * gauss_scale**2
                                Vnonloc[i][
                                    j] = Vnonloc[i][j] + 4. * np.pi * np.sum(
                                        t[:] * VnolRN[k, :] * psiRN[:, j] * w[:]
                                    ) * psiRN[k,
                                              i] * t[k] * w[k] * gauss_scale**2
                            Vnonloc[j][i] = Vnonloc[i][j]
                            Uex[j][i] = Uex[i][j]

                Kin = -mh2 * Kin
                H = Kin + Vloc + zet * Vnonloc
                Hloc = Kin + Vloc
                Norm = U - zet * Uex

                posdef = True
                try:
                    np.linalg.cholesky(Norm)
                except np.linalg.linalg.LinAlgError as err:
                    posdef = False
                posd = '+ definite' if posdef else 'ERROR'

                valnonloc, vecnonloc = scipy.linalg.eig(H[:, :], Norm[:, :])
                zg = np.argsort(valnonloc)
                energiesnonloc = (valnonloc[zg])
                # calculate this only if interaction is non-local, because if it is local,
                # the above will yield that without modification
                # ecce! non-local => energy-dep
                if interaction == 'NonLocal+Local':
                    valloc, vecloc = scipy.linalg.eig(Hloc[:, :], U[:, :])
                    zg = np.argsort(valloc)
                    energiesloc = (valloc[zg])
                elif interaction == 'Local':
                    energiesloc = energiesnonloc

                # calculate eigensystem of the exchange kernel
                # this is done to identify forbidden states one of whose
                # signatures are negative or complex eigenvalues in this kernel
                # ecce! the energy eigenvalues might still be negative and real
                # (peciliarity of generalized EV problems)
                valexkernel, vecexkernel = scipy.linalg.eig(Norm[:, :])

                if interaction == 'NonLocal':
                    print(
                        'A = %d: L = %2.2f , a_core = %2.2f , omega = %e , E(0) = %2.2f + %2.2f i , LeC = %4.4f , LeD = %4.4f, NORM = %s'
                        % (Ncore, Lamb, coreosci, omega,
                           np.real(energiesnonloc[0]),
                           np.imag(energiesnonloc[0]), LeC, LeD, posd))
                    resu.append(
                        np.array([
                            Ncore, Lamb, rad_fac, omega, energiesnonloc[0],
                            LeC, LeD
                        ]))
                else:
                    print(
                        'A = %d: L = %2.2f , a_core = %2.2f , omega = %e , E(0) = %2.2f + %2.2f i , E(0,local) = %2.2f , LeC = %4.4f , LeD = %4.4f, NORM = %s'
                        % (Ncore, Lamb, coreosci, omega,
                           np.real(energiesnonloc[0]),
                           np.imag(energiesnonloc[0]), np.real(energiesloc[0]),
                           LeC, LeD, posd))
                    resu.append(
                        np.array([
                            Ncore, Lamb, rad_fac, omega, energiesnonloc[0],
                            LeC, LeD
                        ]))

                # Check if basis orthonormal:
                if np.sum(abs(np.eye(NState) - U)) > 0.1 * NState**2:
                    print(" ")
                    print("WARNING: omega = ", omega)
                    print("   >>  basis not sufficiently orthonormal: ")
                    print("   >>  average deviation from unit matrix:",
                          np.round(
                              np.sum(abs(np.eye(NState) - U)) / NState**2, 2))
                    #print(np.round(U,2))
                    print("--------------------------")
                    print(" ")
                    print(" ")
                    print(" ")
                    continue

                suff = '%s_%s_%s_%4.4f' % (str(Ncore), str(Lamb), str(rad_fac),
                                           float(omega))
                for compo in [
                        'Unit_loc_%s' % suff,
                        'Unit_ex_%s' % suff,
                        'V_loc_%s' % suff,
                        'V_nonloc_%s' % suff,
                        'E_kin_%s' % suff,
                        'Hamiltonian_%s' % suff
                ]:
                    strFile = compo + '.txt'
                    if os.path.isfile(strFile):
                        os.remove(strFile)
                np.savetxt(
                    'Unit_loc_%s.txt' % suff, np.matrix(U), fmt='%12.4f')
                np.savetxt(
                    'Unit_ex_%s.txt' % suff, np.matrix(Norm), fmt='%12.4f')
                np.savetxt(
                    'V_loc_%s.txt' % suff, np.matrix(Vloc), fmt='%12.4f')
                np.savetxt(
                    'V_nonloc_%s.txt' % suff, np.matrix(Vnonloc), fmt='%12.4f')
                np.savetxt('E_kin_%s.txt' % suff, np.matrix(Kin), fmt='%12.4f')
                np.savetxt(
                    'Hamiltonian_%s.txt' % suff, np.matrix(H), fmt='%12.4f')
                if os.path.isfile('KexEV_%s.dat' % suff):
                    os.remove('KexEV_%s.dat' % suff)
                np.savetxt('KexEV_%s.dat' % suff, valexkernel, fmt='%12.4f')
                if os.path.isfile('HEV_%s.dat' % suff):
                    os.remove('HEV_%s.dat' % suff)
                np.savetxt('HEV_%s.dat' % suff, valnonloc, fmt='%12.4f')

f = plt.figure(figsize=(18, 12))
f.subplots_adjust(left=0.05, right=0.8, bottom=0.1, top=0.82)
f.suptitle(
    r'$\zeta\,H^\Lambda_{non-local}\;\;\;;\zeta=%4.4f\;\;\;;\;\;\;c_{radius}\in[%4.4f\,,\,%4.4f]$'
    % (zet, Aradius_modifyer_fac[0], Aradius_modifyer_fac[-1]),
    fontsize=14)

ax1 = f.add_subplot(111)

ax1.set_xlabel(r'$\omega$ [fm$^{-2}$]', fontsize=12)
ax1.set_ylabel(r'$E_0$ [MeV]', fontsize=12)

for nbrC in range(len(cores)):
    for ll in range(len(Lrange)):
        for cc in range(len(Aradius_modifyer_fac)):

            xx = [
                np.array(resu)[:, 3][ee]
                for ee in range(len(np.array(resu)[:, 3]))
                if ((np.array(resu)[:, 0][ee] == cores[nbrC]) &
                    (np.array(resu)[:, 1][ee] == Lrange[ll]) &
                    (np.array(resu)[:, 2][ee] == Aradius_modifyer_fac[cc]))
            ]

            yy = [
                np.array(resu)[:, 4][ee]
                for ee in range(len(np.array(resu)[:, 4]))
                if ((np.array(resu)[:, 0][ee] == cores[nbrC]) &
                    (np.array(resu)[:, 1][ee] == Lrange[ll]) &
                    (np.array(resu)[:, 2][ee] == Aradius_modifyer_fac[cc]))
            ]

            ax1.plot(
                np.real(xx),
                np.real(yy),
                label=r'A=%d  L=%4.3f' % (cores[nbrC], Lrange[ll]),
                linewidth=4)

            name = r'A=%d  L=%4.3f  c=%4.3f' % (cores[nbrC], Lrange[ll],
                                                Aradius_modifyer_fac[cc])

            y = np.real(yy[-1])

            ax1.annotate(
                name,
                xy=(1, y),
                xytext=(8, 0),
                color=ax1.lines[-1].get_color(),
                xycoords=ax1.get_yaxis_transform(),
                textcoords="offset points",
                size=12,
                va="center")

#plt.legend(loc='best', fontsize=24)
secundo = 3
strFile = 'testNL_%d.pdf' % secundo
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
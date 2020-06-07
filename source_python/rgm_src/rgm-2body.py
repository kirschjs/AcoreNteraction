import os, sys

import numpy as np
from scipy.optimize import fmin

from bridge import *
from rrgm_functions import *
from two_particle_functions import *
from C0D1_lec_sets import *

#                      result result
#              J  S CH [e,ph] [lec]  label color reduce? scale
chan = 'np-3S1'
chs_nn = {
    # DEUTERON
    'np-3SD1': [1, 1, [2, 10]],
    'np-3S1': [1, 1, [2]],
    'np-1S0': [0, 0, [1]],
    'np-1P1': [1, 0, [8]],
    'np-3P0': [0, 1, [9]],
    'np-3P1': [1, 1, [9]],
    'np-3P2': [2, 1, [9]],
    'nn-1S0': [0, 0, [4]],
    'nn-3P0': [0, 1, [5]],
    'nn-3P1': [1, 1, [5]],
    'nn-3P2': [2, 1, [5]],
    # NEUTERON
    'nn-3PF2': [2, 1, [5, 7]],
    'pp-1S0': [0, 0, [3]],
    'nn-3F2': [2, 1, [7]],
    'pp-3P0': [0, 1, [12]],
    'pp-3P1': [1, 1, [12]],
    'pp-3P2': [2, 1, [12]]
}

nn_phases = []
nn_scatlengths = []
dimerBDGs = []
dd_phases = []
dd_scatlengths = []
alphs = []
lams = []

nn_scale = 1.55
dd_scale = 1.25
addw = 0

plo = 0
verb = 0

#  lec_list = [
#      line for line in open(
#          '/home/kirscher/kette_repo/AcoreNteraction/source_mathematica/tmp.dat')
#  ]

lec_list = C0D1_lec_set['tmp']

print(
    'L[fm^-1]   alpha[fm^-2]    C0[MeV]        B_d[MeV]   a_ff[fm]   a_dd[fm]')

head = 'L[fm^-1]   alpha[fm^-2]    C0[MeV]        B_d[MeV]   a_ff[fm]   a_dd[fm]\n'

for lec in lec_list:

    lam = float(lec)

    n2path = home + '/kette_repo/sim_par/nucleus/2n/%s/' % str(lam)
    if os.path.isdir(n2path) == False:
        os.system('mkdir ' + n2path)
    if os.path.isfile(n2path + 'INQUA_N') == True:
        os.system('rm ' + n2path + 'INQUA_N')

    os.chdir(n2path)
    h2_inlu(anzo=7)
    os.system(BINpath + 'LUDW_EFT_new.exe')
    h2_inob(anzo=5)
    os.system(BINpath + 'KOBER_EFT_nn.exe')
    phasSeum = [np.zeros(anze) for n in range(anzs)]

    potnn = 'pot_nn_%5s' % str(lam)
    potdd = 'pot_dd_%5s' % str(lam)

    cloW = float(lec_list[lec][0])
    cloB = 0.

    prep_pot_files([0.25 * lam**2], [cloW], [], [], [], potnn)

    jay = chs_nn[chan][0]
    stot = chs_nn[chan][1]
    tmp1 = []
    tmp2 = []
    coeffnn = np.array(
        #  coul, cent,p^2,r^2,            LS,        TENSOR, TENSOR_p
        [1., 1., 0, 0, 0, 0, 0])
    costrnn = ''
    for fac in coeffnn:
        costrnn += '%12.6f' % float(fac)
    rw = wid_gen(add=addw, w0=w120, ths=[1e-5, 3e2, 0.2], sca=nn_scale)
    h2_inqua(rw, potnn)
    os.system(BINpath + 'QUAFL_' + mpii + '.exe')
    h2_inen_bs(relw=rw, costr=costrnn, j=jay, ch=chs_nn[chan][2])
    os.system(BINpath + 'DR2END_' + mpii + '.exe')
    Bnn = get_h_ev()[0]
    h2_inen_str_pdp(relw=rw, costr=costrnn, j=jay, sc=stot, ch=chs_nn[chan][2])
    os.system(BINpath + 'DR2END_' + mpii + '.exe')
    h2_spole(
        nzen=anze,
        e0=0.0001,
        d0=0.001,
        eps=0.01,
        bet=2.1,
        nzrw=400,
        frr=0.06,
        rhg=8.0,
        rhf=1.0,
        pw=0)
    os.system(BINpath + 'S-POLE_PdP.exe')
    nn_phases.append(
        read_phase(phaout='PHAOUT', ch=[1, 1], meth=1, th_shift=''))
    e0 = nn_phases[-1][0][0]
    d0 = nn_phases[-1][0][2]
    nn_scatlengths.append(-(float(e0) * mn[mpii] / MeVfm**2)**
                          (-0.5) * np.tan(float(d0) * np.pi / 180))

    dimerBDGs.append(get_h_ev()[0])

    #alph = 1.5 * nn_scatlengths[-1]**(-2)  # [] = fm^-2
    alph = 1.5 * np.abs(dimerBDGs[-1]) * mn['137'] / MeVfm**2
    alphs.append(alph)
    lams.append(lam)

    prefac = MeVfm**2 / (2. * mn['137'])  # [] = MeV fm^2

    cW = prefac * 2. * (alph)  # [] = MeV
    rW = prefac * 16. * (alph)**2  # [] = MeV fm^-2

    LECr2 = 8 * np.pi**1.5 * 4 * alph**2 * prefac
    EXPr2 = alph

    LECatt0 = -2 * prefac * alph
    EXPatt0 = alph

    LECrep = -8 * np.pi**1.5 * 2 * cloW * (2 * alph / (2 * alph + lam))**1.5
    EXPrep = alph * (2 * alph + 3 * lam) / (2 * alph + lam)

    LECatt = 2 * cloW * (2 * alph / (2 * alph + lam))**1.5
    EXPatt = (2 * alph * lam) / (2 * alph + lam)

    prep_pot_files([EXPatt0, EXPatt, EXPrep, EXPr2], [LECatt0, LECatt, LECrep],
                   [], [LECr2], [], potdd)

    coeffdd = np.array(
        #  coul, cent,p^2,r^2,            LS,        TENSOR, TENSOR_p
        [1, 1, 0, 1, 0, 0, 0.])
    costrdd = ''
    for fac in coeffdd:
        costrdd += '%12.6f' % float(fac)
    rw = wid_gen(add=addw, w0=w120, ths=[1e-5, 2e2, 0.2], sca=dd_scale)
    h2_inqua(rw, potdd)
    os.system(BINpath + 'QUAFL_DD.exe')
    h2_inen_bs(relw=rw, costr=costrdd, j=jay, ch=chs_nn[chan][2])
    os.system(BINpath + 'DR2END_DD.exe')
    Bdd = get_h_ev()[0]
    h2_inen_str_pdp(relw=rw, costr=costrdd, j=jay, sc=stot, ch=chs_nn[chan][2])
    os.system(BINpath + 'DR2END_DD.exe')
    os.system(BINpath + 'S-POLE_PdP.exe')
    dd_phases.append(
        read_phase(phaout='PHAOUT', ch=[1, 1], meth=1, th_shift=''))
    e0 = dd_phases[-1][0][0]
    d0 = dd_phases[-1][0][2]

    dd_scatlengths.append(-(float(e0) * 2 * mn[mpii] / MeVfm**2)**
                          (-0.5) * np.tan(float(d0) * np.pi / 180))

    print('%2.4f     %8.8f      %8.6f      %4.2f      %4.2f      %4.2f' %
          (lam, alph, cloW, dimerBDGs[-1], nn_scatlengths[-1],
           dd_scatlengths[-1]))

    head += '%2.4f     %8.8f      %8.6f      %4.2f      %4.2f      %4.2f\n' % (
        lam, alph, cloW, dimerBDGs[-1], nn_scatlengths[-1], dd_scatlengths[-1])
    #print(
    #    'L = %2.2f fm^-1     B(NN) = %4.4f MeV     a(NN) = %4.4f fm       a_RGM = %8.8f       B(DD) = %4.4f MeV     a(DD) = %4.4f fm'
    #    % (lam, Bnn, nn_scatlengths[-1], alph, Bdd, dd_scatlengths[-1]))

outf = '/home/kirscher/kette_repo/AcoreNteraction/manuscript/lambda_alpha_%1.4f_C0_Bff_aff_%2.1f_a_DD.dat' % (
    np.mean(alphs), np.mean(nn_scatlengths))
with open(outf, 'w') as outfile:
    outfile.write(head)

print('data exported to:%s' % outf)

fig = plt.figure(figsize=(10, 6), dpi=95)

fig.suptitle(
    r'$V_{d-d}=\lim_{\Lambda/a_{nn}\to\infty}V_{d-d}^{(rgm)}=-\frac{\hbar^2}{2\mu}(16\,\alpha^2\,R^2+2\alpha)e^{-\alpha\,R^2}\;\;\;,\;\;\;\alpha\in[%4.6f,%4.6f]fm\;\;\;,\;\;\;E_0(NN)=%2.2fMeV$'
    % (alphs[0], alphs[-1], dimerBDGs[0]))

ax1 = plt.subplot(121)
ax1.set_xlabel(r'$E_{cm}\;\;[MeV]$', fontsize=16)
ax1.set_ylabel(r'$\delta_0\;\; [deg]$', fontsize=16)
[
    ax1.plot(
        [ph[0] for ph in phset], [ph[2] for ph in phset],
        label=r'$nucleon-nucleon$',
        color='black',
        linestyle='--',
        linewidth=1) for phset in nn_phases
]
[
    ax1.plot(
        [ph[0] for ph in phset], [ph[2] for ph in phset],
        label=r'$dimer-dimer$',
        color='gray',
        linewidth=1) for phset in dd_phases
]
leg = ax1.legend(loc='best')

ax1 = plt.subplot(122)
ax1.set_xlabel(r'$\Lambda\;\;[fm^{-1}]$', fontsize=16)
ax1.set_ylabel(r'', fontsize=16)
ax1.set_ylim(-2, 2)

ax1.plot(
    lams,
    np.array(dd_scatlengths) / np.array(nn_scatlengths),
    label=r'$a_{dd}/a_{nn}$',
    color='black',
    linewidth=1)
leg = ax1.legend(loc='best')

outstr = '/home/kirscher/kette_repo/AcoreNteraction/manuscript/NN_dimer-dimer_vergleich.pdf'
fig.savefig(outstr)
print('results are in %s' % outstr)
#plt.show()
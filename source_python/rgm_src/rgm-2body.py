import os, sys

import numpy as np
from scipy.optimize import fmin

from bridge import *
from rrgm_functions import *
from two_particle_functions import *

#                      result result
#              J  S CH [e,ph] [lec]  label color reduce? scale
chs = {
    # DEUTERON
    #'np-3SD1': [1, 1, [2, 10], [], [], 'SD', 'blue', sizeFrag, 1.0],
    'np-3S1': [1, 1, [2], [], [], 'S', 'blue', sizeFrag, 1.21],
    'np-1S0': [0, 0, [1], [], [], 'S', 'red', sizeFrag, 1.32],
    #'np-1P1': [1, 0, [8], [], [], 'P', 'gray', sizeFrag, 1.0],
    #'np-3P0': [0, 1, [9], [], [], 'P', 'red', sizeFrag, 0.95],
    #'np-3P1': [1, 1, [9], [], [], 'P', 'green', sizeFrag, 0.98],
    #'np-3P2': [2, 1, [9], [], [], 'P', 'blue', sizeFrag, 1.0],
    #'nn-1S0': [0, 0, [4], [], [], 'S', 'orange', sizeFrag, 0.9],
    #'nn-3P0': [0, 1, [5], [], [], 'P', 'red', sizeFrag, 0.81],
    #'nn-3P1': [1, 1, [5], [], [], 'P', 'green', sizeFrag, .99],
    #'nn-3P2': [2, 1, [5], [], [], 'P', 'blue', sizeFrag, 0.95]
    # NEUTERON
    #'nn-3PF2': [2, 1, [5, 7], [], [], 'PF', 'blue'],
    #'pp-1S0': [0, 0, [3], [], [], 'S', 'red', sizeFrag, 2.0]
    #'nn-3F2': [2, 1, 7, [], [], 'F']
    #'pp-3P0': [0, 1, 12, [], [], 'P', 'red'],
    #'pp-3P1': [1, 1, 12, [], [], 'P', 'green'],
    #'pp-3P2': [2, 1, 12, [], [], 'P', 'blue']
}

scale = 1.3
addw = 10

plo = 0
verb = 0

optLECs = {}

for lam in Lrange:
    print('L = %2.2f' % (lam))
    la = ('%-4.2f' % lam)[:4]
    n2path = home + '/kette_repo/sim_par/nucleus/2n/' + la + '/'

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
    pots = 'pot_nn_%02d' % int(float(la))
    valinter = False
    optlist = False
    try:
        lec_list = lec_list_c['137']
        optlist = True
    except:
        print('no LEC list loaded!')
        exit()

    try:
        cloW = 0.5 * (lec_list[la][0] + lec_list[la][1])
        cloB = 0.5 * (lec_list[la][0] - lec_list[la][1])
    except:
        valinter = True
        xy = np.array([[float(l), lec_list[l][0]] for l in lec_list.keys()])
        x = xy[:, 0]
        y = xy[:, 1]
        z = np.polyfit(x, y, 3)
        p = np.poly1d(z)
        cloW = p(lam)
        cloB = 0
        print('>>> interpolated LECs for this cutoff: c0 = %4.4f MeV' % cloW)

    #                       wiC                  baC    wir2  bar2    ls  ten
    prep_pot_files_pdp(lam, cloW, cloB, r2w * cloW, r2b * cloB, ls * cloW,
                       ten * cloW, pots)

    for chan in chs:
        jay = chs[chan][0]
        stot = chs[chan][1]

        tmp1 = []
        tmp2 = []

        for n in range(len(eps_space)):

            if (('bdg' in cal) | ('reduce' in cal)):
                coeff = np.array(
                    #  coul, cent,p^2,r^2,            LS,        TENSOR, TENSOR_p
                    [1., pot_scale * eps_space[n], 0., 0, 0, 0, 0.])
                costr = ''
                for fac in coeff:
                    costr += '%12.6f' % float(fac)
                rw = wid_gen(
                    add=addw, w0=w120, ths=[1e-5, 2e2, 0.2], sca=chs[chan][8])
                h2_inqua(rw, pots)
                os.system(BINpath + 'QUAFL_' + mpii + '.exe')
                h2_inen_bs(relw=rw, costr=costr, j=jay, ch=chs[chan][2])
                os.system(BINpath + 'DR2END_' + mpii + '.exe')

                if dbg:
                    print(
                        'LS-scheme: B(2,%s,eps=%2.2f) = %4.4f MeV [' %
                        (chan, eps_space[n], get_h_ev()[0]),
                        get_h_ev(n=4),
                        ']')
                rrgm_functions.parse_ev_coeffs()
                os.system('cp OUTPUT end_out_b && cp INEN inen_b')

            if 'reduce' in cal:

                reduce_2n(
                    w2rels=rw,
                    ch=chan,
                    size2=sizeFrag,
                    ncycl=ncycl,
                    maxd=maxDiff,
                    minc2=minCoef,
                    maxc2=maxCoef)
                if dbg:
                    print('-- reduced B(2,%s) = %4.4f MeV' % (chan,
                                                              get_h_ev()[0]))

                os.system(
                    'cp OUTPUT end_out_b && cp INEN INEN_%s && cp INQUA_N INQUA_N_%s'
                    % (chan, chan))
                os.system('cp ' + 'COEFF ' + 'COEFF_' + chan)
                os.system('cp ' + 'OUTPUT ' + 'out_' + chan)

            if 'over' in cal:
                for Lov in over_space:
                    for pair in ['singel', 'tripl']:
                        kplME = overlap(
                            bipa=BINpath,
                            chh=chan,
                            Lo=Lov,
                            pair=pair,
                            mpi=mpii)
                        print('< %s | (%s-contact,L=%4.4f) | %s > = %4.4e' %
                              (chan, pair[:4], Lov, chan, kplME))

            if 'scatt' in cal:
                h2_inen_str_pdp(
                    relw=rw, costr=costr, j=jay, sc=stot, ch=chs[chan][2])
                os.system(BINpath + 'DR2END_' + mpii + '.exe')
                if verb: os.system('cp OUTPUT end_out_s && cp INEN inen_s')
                h2_spole(
                    nzen=anze,
                    e0=0.001,
                    d0=0.5,
                    eps=0.01,
                    bet=2.1,
                    nzrw=400,
                    frr=0.06,
                    rhg=8.0,
                    rhf=1.0,
                    pw=0)
                os.system(BINpath + 'S-POLE_PdP.exe')
                if verb:
                    os.system('cp PHAOUT pho_j=%d_l=%s_e=%f' % (jay, lam,
                                                                eps_space[n]))

                for cr in range(1, len(chs[chan][5]) + 1):
                    for cl in range(1, cr + 1):

                        phases = read_phase(
                            phaout='PHAOUT', ch=[cl, cr], meth=1, th_shift='')
                        chs[chan][3].append(phases)
                        phasSeum[n] += np.array(phases)[:, 2]
                        chs[chan][4].append(eps_space[n])
                        print(
                            'a(NN,%s,eps=%2.2f) = %17f fm\n' %
                            (chan, eps_space[n],
                             -(float(chs[chan][3][-1][-1][0]
                                     ) * mn[mpii] / MeVfm**2)**(-0.5) * np.
                             tan(float(chs[chan][3][-1][-1][2]) * np.pi / 180)
                             ),
                            end='\n')

    if 'plot' in cal:
        if 'scatt' not in cal:
            print('ECCE >>> plotting phases without a scattering calc.')

        if dbg == 2:
            try:
                for n in range(len(phasSeum[-1])):
                    print('%4.4f  %4.4f  %4.4f  %4.4f' %
                          (phasSeum[-1][n], float(chs['nn-3P0'][3][0][n][2]),
                           float(chs['nn-3P1'][3][0][n][2]),
                           float(chs['nn-3P2'][3][0][n][2])))
            except:
                print('no P-waves')
            for lec in range(len(eps_space)):
                print('%2.1f%12.6f ' % (lam, eps_space[lec]), end='')
                for ch in chs:
                    print(
                        '%17f' %
                        (-(float(chs[ch][3][lec][-1][0]) * mn[mpii] / MeVfm**2
                           )**(-1.5) * np.tan(
                               float(chs[ch][3][lec][-1][2]) * np.pi / 180)),
                        end='')
                print('')

        fig = plt.figure()
        chss = []
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xlabel(r'$E_{cm}\;\;[MeV]$', fontsize=21)
        ax1.set_ylabel(r'$\delta(nn)\;\; [deg]$', fontsize=21)
        ax1.set_title(
            r'$\Lambda=%d fm^{-1}:\;\;\;\;c(LS)=%2.3f\;\;\;\;\;c(S_{12})=%2.3f$'
            % (int(lam), ls * eps_space[-1], ten * eps_space[-1]),
            fontsize=21)

        if phasSeum:
            ax1.plot(
                [ph[0] for ph in chs[chan][3][-1]],
                phasSeum[0],
                label=r'$\sum_\delta(\epsilon=0)$',
                color='black')
            ax1.plot(
                [ph[0] for ph in chs[chan][3][-1]],
                phasSeum[-1],
                label=r'$\sum_\delta(\epsilon=max.)$',
                color='black',
                linestyle='dotted')

        nn = 1

        for chan in chs:

            ax1.plot(
                [ph[0] for ph in chs[chan][3][0]
                 ][:int(len(chs[chan][3][0]) * (10 - nn) / 10)], [
                     ph[2] for ph in chs[chan][3][0]
                 ][:int(len(chs[chan][3][0]) * (10 - nn) / 10)],
                label=r'$C_2(%s)=%.2E$' % (chan, chs[chan][4][0]),
                color=chs[chan][-3],
                linewidth=5 - nn)
            ax1.plot(
                [ph[0] for ph in chs[chan][3][-1]],
                [ph[2] for ph in chs[chan][3][-1]],
                label=r'$C_2(^%1d%s_%1d)=%.2E$' % (int(2 * chs[chan][1] + 1),
                                                   chs[chan][-4], chs[chan][0],
                                                   chs[chan][4][-1]),
                color=chs[chan][-3],
                linestyle='dashed',
                linewidth=2,
                alpha=0.5)
            nn += 1
            [
                ax1.plot(
                    [ph[0] for ph in chs[chan][3][-1]],
                    [ph[2] for ph in chs[chan][3][n]],
                    color=chs[chan][-3],
                    alpha=0.25,
                    lw=1) for n in range(1,
                                         len(chs[chan][3]) - 1)
            ]
            leg = ax1.legend(loc='best')
            #plt.ylim(-.1, .1)

        plt.show()
        exit()

    if 'fit' in cal:

        def fitti(fac2, fitb, fix=0, blabla=0):
            repl_line('INEN', 2, '%12.6f%12.6f%12.6f%12.6f%12.6f\n' %
                      (1.0, fac2, 0.0, 0.0, 0.0))
            os.system(BINpath + 'DR2END_' + mpii + '.exe')
            lines_output = [line for line in open('OUTPUT')]
            for lnr in range(0, len(lines_output)):
                if lines_output[lnr].find(
                        'EIGENWERTE DES HAMILTONOPERATORS') >= 0:
                    E_0 = lines_output[lnr + 3].split()[fix]
            return abs(float(E_0) + fitb)

        print('>>> commencing fit...')

        deub = 1.0

        fac = 1.001
        ft_lo = fmin(fitti, fac, args=(deub, 0, 1), disp=False)
        res_lo = fitti(ft_lo[0], 0.0, 0, 0)
        print('L = %2.2f:  %12.4f yields B(2)= %8.4f' % (lam, cloW * ft_lo[0],
                                                         res_lo))
        optLECs[la] = [cloW * ft_lo[0], 0.0]
        prep_pot_files_pdp(lam, cloW * ft_lo[0], 0., 0., 0., 0., 0., pots)

if 'fit' in cal:
    print(optLECs)
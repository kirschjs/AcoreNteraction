from pathlib import Path
home = str(Path.home())

import os, re
import numpy as np
import random
import rrgm_functions
import LEC_fit_setup


def red_mod_2(
        max_coeff=11000,
        min_coeff=150,
        target_size=7,
        nbr_cycles=20,
        max_diff=0.01,
        ord=0,
        dr2executable=home +
        '/kette_repo/rrgm/source/seriell/eft_sandkasten/DR2END_AK_I_2.exe'):

    bdg_end = 0.0
    basis_size = 400000
    diff = 0.0
    nc = 0
    while (nc <= nbr_cycles) & (basis_size > target_size):
        #while (basis_size>target_size):
        # print currently lowest eigenvalue
        lines_output = [line for line in open('OUTPUT')]
        for lnr in range(0, len(lines_output)):
            if lines_output[lnr].find('EIGENWERTE DES HAMILTONOPERATORS') >= 0:
                bdg_ini = float(lines_output[lnr + 3].split()[ord])
        #print('Initial binding energy: B(3) = %f MeV' % (bdg_ini))

        # read file OUTPUT
        bv_ent = []
        for lnr in range(0, len(lines_output)):
            if lines_output[lnr].find(
                    'ENTWICKLUNG DES  %1d TEN EIGENVEKTORS,AUSGEDRUECKT DURCH NORMIERTE BASISVEKTOREN'
                    % (ord + 1)) >= 0:
                for llnr in range(lnr + 2, len(lines_output)):
                    if lines_output[llnr] == '\n':
                        break
                    else:
                        try:
                            if (int(lines_output[llnr].split(')')[0]) !=
                                    len(bv_ent) + 1):
                                bv_ent[-1] += lines_output[llnr][
                                    lines_output[llnr].find(')') + 1:].rstrip(
                                    )[2:]
                            else:
                                bv_ent.append(
                                    lines_output[llnr][lines_output[llnr].find(
                                        ')') + 1:].rstrip()[2:])
                        except:
                            continue
                            #print( 'EOF.')
        # identify the vectors with insignificant contribution;
        # the result is a pair (bv number, {relw1, relw2, ...})
        bv_to_del = []
        basis_size = 0
        for nn in bv_ent:
            basis_size += len(nn) / 8

        #print(bv_ent, basis_size)

        for bv in range(1, len(bv_ent) + 1):
            relw_to_del = []
            tmpt = bv_ent[bv - 1]
            ueco = [
                tmpt[8 * n:8 * (n + 1)]
                for n in range(0, int((len(tmpt.rstrip())) / 8))
            ]
            ueco = [tmp for tmp in ueco if (tmp != '') & (tmp != '\n')]
            for coeff in range(0, len(ueco)):
                try:
                    if (abs(int(ueco[coeff])) > max_coeff) | (abs(
                            int(ueco[coeff])) < min_coeff):
                        relw_to_del.append(coeff)
                except:
                    relw_to_del.append(coeff)
            try:
                bv_to_del.append([bv, relw_to_del])
            except:
                print('bv %d is relevant!' % bv)
        rednr = sum([len(tmp[1]) for tmp in bv_to_del])
        if rednr == 0:
            print('All abnormally large/small BV were removed.')
            break
        #if (len(bv_ent[0])/8==target_size):
        #   #os.system('cp inen_bkp INEN')
        #   print( 'target size (%d) reached. ' %int(len(bv_ent[0])/8))
        #   break
        #   # from the input file INEN remove the basis vectors with
        #   # number bv=bv_to_del[0] and relative widths from the set bv_to_del[1]
        #   # note: the indices refer to occurance, not abolute number!
        #   # e.g.: bv is whatever vector was included in INEN as the bv-th, and the
        #   # rel-width is the n-th calculated for this bv

        lines_inen = [line for line in open('INEN')]
        bv_to_del = [tmp for tmp in bv_to_del if tmp[1] != []]
        #print(bv_to_del)
        random.shuffle(bv_to_del)
        to_del = 1
        # 1. loop over all bv from which relw can be deleted
        for rem in bv_to_del[:max(1, min(to_del, len(bv_to_del) - 1))]:
            ll = ''
            # 2. calc line number in INEN where this vector is included
            repl_ind = 4 + 2 * (rem[0])
            # repl_ind = 8
            repl_line = lines_inen[repl_ind - 1]
            repl_ine = []
            #
            random.shuffle(rem[1])
            for rel_2_del in rem[1]:
                #print( 'removing relw %d' %rel_2_del)
                for relnr in range(0, len(repl_line.split())):
                    if int(repl_line.split()[relnr]) == 1:
                        occ = 0
                        for tt in repl_line.split()[:relnr + 1]:
                            occ += int(tt)
                        if occ == rel_2_del + 1:
                            repl_ine.append(relnr)
                break

            ll = ''
            for relnr in range(0, len(repl_line.split())):
                repl = False
                if int(repl_line.split()[relnr]) == 1:
                    for r in repl_ine:
                        if relnr == r:
                            repl = True
                            pass
                    if repl:
                        ll += '  0'
                    else:
                        ll += '%+3s' % repl_line.split()[relnr]
                else:
                    ll += '%+3s' % repl_line.split()[relnr]
            ll += '\n'

            lines_inen[repl_ind - 1] = ll

        s = ''
        for line in lines_inen:
            s += line

        os.system('cp INEN inen_bkp')
        with open('INEN', 'w') as outfile:
            outfile.write(s)

        os.system(dr2executable)
        os.system('cp OUTPUT out_bkp')
        lines_output = [line for line in open('OUTPUT')]
        for lnr in range(0, len(lines_output)):
            if lines_output[lnr].find('EIGENWERTE DES HAMILTONOPERATORS') >= 0:
                bdg_end = float(lines_output[lnr + 3].split()[ord])
        diff = abs(bdg_end - bdg_ini)
        #print('%2d:B(2,%d)=%f || B(red)-B = %f' % (nc, basis_size - 1, bdg_end,
        #                                           diff), )
        if (diff > max_diff):
            #print('B(red)-B > maxD')
            os.system('cp inen_bkp INEN')
            os.system('cp out_bkp OUTPUT')
        nc = nc + 1
    return bdg_end, basis_size


def reduce_2n(w2rels,
              ch='nn-1S0',
              size2=20,
              ncycl=50,
              maxd=0.01,
              minc2=500,
              maxc2=3000):
    cons_red = 1
    print('reducing widths in %s channel...' % ch)
    while cons_red:
        tmp = red_mod_2(
            max_coeff=maxc2,
            min_coeff=minc2,
            target_size=size2,
            nbr_cycles=ncycl,
            max_diff=maxd,
            ord=0,
            dr2executable=LEC_fit_setup.BINpath + 'DR2END_' +
            LEC_fit_setup.mpii + '.exe')
        os.system('cp ' + 'INEN ' + 'INEN_' + ch)

        mm_s = rrgm_functions.get_reduced_width_set(
            w2rels, 'INEN_' + ch, bsorsc='b', outf='WIDTH_' + ch)

        os.system('cp ' + 'INQUA_N ' + 'INQUA_N_' + ch)
        rrgm_functions.parse_ev_coeffs()
        os.system('cp ' + 'COEFF ' + 'COEFF_' + ch)
        os.system('cp ' + 'OUTPUT ' + 'out_' + ch)
        tmpa = sum([
            int(i)
            for i in [line for line in open('INEN_' + ch)][5].strip().split()
        ])
        tmpb = len([line for line in open('COEFF_' + ch)])

        cons_red = ((tmpa != tmpb) | (size2 <= int(tmp[1])))
        minc2 += 10
        print(minc2, tmp[1], size2, tmpa, tmpb)

    print('reduction to %d widths complete.' % size2)


def h2_inen_str_pdp(relw, costr, j=0, sc=0, ch=[1]):
    s = ''
    s += ' 10  2 12  9  1  1 -0  0  0 -1\n'
    s += '  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n'
    #
    s += '%s\n' % costr
    #     2*J #ch s/b
    s += '%4d%4d   0   0   2\n' % (int(2 * j), len(ch))
    for c in ch:
        s += '  1  1%3d\n' % int(2 * sc)
        s += '   1%4d\n' % int(c)
        for rr in range(1, 24):
            s += '  1'
        s += '\n'
    with open('INEN', 'w') as outfile:
        outfile.write(s)
    return


def h2_spole(nzen=20,
             e0=0.01,
             d0=0.075,
             eps=0.01,
             bet=1.1,
             nzrw=400,
             frr=0.06,
             rhg=8.0,
             rhf=1.0,
             pw=1):
    s = ''
    s += ' 11  3  0  0  0  1\n'
    s += '%3d  0  0\n' % int(nzen)
    s += '%12.4f%12.4f\n' % (float(e0), float(d0))
    s += '%12.4f%12.4f%12.4f\n' % (float(eps), float(eps), float(eps))
    s += '%12.4f%12.4f%12.4f\n' % (float(bet), float(bet), float(bet))
    #    OUT
    s += '  0  0  1  0  1  0 -0  0\n'
    s += '%3d\n' % int(nzrw)
    s += '%12.4f%12.4f%12.4f\n' % (float(frr), float(rhg), float(rhf))
    s += '  1  2  3  4\n'
    s += '0.0         0.0         0.0\n'
    s += '.001        .001        .001\n'
    if pw == 0:
        s += '.5          .5          .5          .5\n'
    elif pw == 1:
        s += '.3          .3          .3          .3\n'
    elif pw == 2:
        s += '.15         .15         .15         .15\n'
    s += '1.          1.          0.\n'
    with open('INPUTSPOLE', 'w') as outfile:
        outfile.write(s)
    return


def h2_inlu(anzo=5):
    s = ''
    s += '  9\n'
    for n in range(anzo):
        s += '  1'
    s += '\n  1\n'
    s += '  4  2\n'
    s += '  0\n  1\n  2\n  3'
    with open('INLUCN', 'w') as outfile:
        outfile.write(s)
    return


def h2_inob(anzo=5):
    s = ''
    s += '  0  0\n'
    for n in range(anzo):
        s += '  1'
    s += '\n  4\n'
    s += '  1  2\n'
    s += '  2  9  6  1\n'
    s += '  1  1\n'
    s += '  1  3\n'  #  p-up, n-up
    s += '  1  4\n'  #  ...
    s += '  2  3\n'
    s += '  1  2\n'
    s += '  2  1\n'
    s += '  3  4\n'
    s += '  4  3\n'
    s += '  3  3\n'  # n-up, n-up
    s += '  1  1\n'  # p-up, p-up
    s += '  1  1\n'
    s += '  0  1  1  2\n'
    s += '  0  1 -1  2\n'
    s += '  0  1  0  1  1  2\n'
    s += '  0  1  0  1 -1  2\n'
    s += '  0  1  0  1  0  1  1  2\n'
    s += '  0  1  0  1  0  1 -1  2\n'
    s += '  0  1  0  1  0  1  0  1  1  1\n'
    s += '  0  1  0  1  0  1  0  1  0  1  1  1'
    with open('INOB', 'w') as outfile:
        outfile.write(s)
    return


def h2_inqua(relw, ps2):
    s = ''
    s += ' 10  8  9  3 00  0  0  0  0\n'
    #s += pot_dir + ps2 + '\n'
    s += ps2 + '\n'
    s += ' 14\n'
    s += '  1%3d\n' % int(len(relw))
    s += '.0          .0\n'
    for relwl in range(0, int(np.ceil(float(len(relw)) / float(6)))):
        for rr in range(0, 6):
            if (relwl * 6 + rr) < len(relw):
                s += '%12.7f' % float(relw[relwl * 6 + rr])
        s += '\n'
    s += '  2  1\n1.\n'  # 1:  n-p 1S0
    s += '  1  1\n1.\n'  # 2:  n-p 3S1
    # ------------
    s += '  3  1\n1.\n'  # 3:  p-p 1S0
    # ------------
    s += '  4  1\n1.\n'  # 4:  n-n 1S0
    s += '  5  2\n1.\n'  # 5:  n-n 3P0,1,2
    s += '  4  3\n1.\n'  # 6:  n-n 1D2
    s += '  5  4\n1.\n'  # 7:  n-n 3F2,3,4
    # ------------
    s += '  2  2\n1.\n'  # 8:  n-p 1P1
    s += '  1  2\n1.\n'  # 9:  n-p 3P0,1,2
    s += '  1  3\n1.\n'  # 10: n-p 3D1
    # ------------
    s += '  3  1\n1.\n'  # 4:  p-p 1S0
    s += '  6  2\n1.\n'  # 5:  p-p 3P0,1,2
    s += '  3  3\n1.\n'  # 6:  p-p 1D2
    s += '  6  4\n1.'  # 7:  p-p 3F2,3,4
    with open('INQUA_N', 'w') as outfile:
        outfile.write(s)
    return
    # r7 c2:   S  L           S_c
    #  1   :   0  0  1S0         0
    #  2   :   1  0  3S1         2
    #  3   :   0  0  1S0         0          p-p
    #  4   :   0  0  1S0         0          n-n
    #  5   :   1  1  3P0,3P1,3P2 2          n-n
    #  6   :   0  2  1D2         2          n-n
    #  7   :   0  1  1P1         0
    #  8   :   1  1  3P0,3P1,3P2 2
    #  9   :   1  2  3D1         2


def h2_inen_bs(relw, costr, j=0, ch=1, anzo=9, nzz=0, EVein=[]):
    s = ''
    s += ' 10  2 12%3d  1  1%3d  0  0 -1\n' % (int(anzo), nzz)
    #       N  T Co CD^2 LS  T
    s += '  1  1  1  1  1  1  1  1  1  1\n'

    s += '%s\n' % costr

    #     2*J #ch s/b
    s += '%4d%4d   1   0   2\n' % (int(2 * j), len(ch))
    for c in ch:
        s += '   1%4d\n' % int(c)
        for rr in range(len(relw)):
            s += '  1'
        s += '\n'

    if nzz < 0:
        for c in EVein:
            s += c

    with open('INEN', 'w') as outfile:
        outfile.write(s)
    return


def h2_inen_str(relw, costr, j=0, sc=0, ch=1, anzo=7):
    s = ''
    s += ' 10  2 12%3d  1  1 -0  0  0 -1\n' % int(anzo)
    #      N  T Co  CD^2 LS  T
    s += '  1  1  1  1  1  1  1  1  1  1\n'
    #
    s += '%s\n' % costr
    #     2*J #ch s/b
    s += '%4d   1   0   0   2\n' % int(2 * j)
    s += '  1  1%3d\n' % int(2 * sc)
    s += '   1%4d\n' % int(ch)
    for rr in range(1, len(relw) + 1):
        if ((rr % 30 == 0)):
            s += '  1'
            s += '\n'
        s += '  1'
    with open('INEN', 'w') as outfile:
        outfile.write(s)
    return
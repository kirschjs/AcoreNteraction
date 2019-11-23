import os, re
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import random
from parameters_and_constants import *


def get_h_ev(n=1, ifi='OUTPUT'):
    out = [line for line in open(ifi)]
    for nj in range(1, len(out)):
        if (out[nj].strip() == "EIGENWERTE DES HAMILTONOPERATORS"):
            E_0 = out[nj + 3].split()
    return np.array(E_0[:n]).astype(float)


def get_bind_en(n=1, ifi='OUTPUT'):
    out = [line for line in open(ifi)]
    for nj in range(1, len(out)):
        if (out[nj].strip() == "BINDUNGSENERGIEN IN MEV"):
            E_0 = out[nj + 1].split()
            break
    return np.array(E_0[:n]).astype(float)


def get_kopplungs_ME(op=4, ifi='OUTPUT'):

    out = [line for line in open(ifi)]

    for nj in range(1, len(out)):
        if (out[nj].strip() == 'KOPPLUNGSMATRIX FUER OPERATOR    %d' % op):
            E_0 = out[nj + 1].split()[0].strip()

    return float(E_0)


def overlap(bipa, chh, Lo=6.0, pair='singel', mpi='137'):

    prep_pot_files_pdp(Lo, 2.0, (-1)**(len(pair) % 2 - 1) * 2.0, 0.0, 0.0, 0.0,
                       0.0, 'pot_' + pair)
    repl_line('INQUA_N', 1, 'pot_' + pair + '\n')

    os.system(bipa + 'QUAFL_' + mpi + '.exe')

    uec = [line for line in open('COEFF')]
    s = ''
    for a in uec:
        s += a

    os.system('cp inen_b INEN')

    repl_line('INEN', 0, ' 10  2 12  9  1  1 -1  0  0 -1\n')
    repl_line('INEN', 2, '%12.6f%12.6f%12.6f\n' % (float(1.), float(1.),
                                                   float(0.)))

    with open('INEN', 'a') as outfile:
        outfile.write(s)

    os.system(bipa + 'DR2END_' + mpi + '.exe')

    os.system('cp OUTPUT end_out_over_' + pair + '_' + chh +
              ' && cp INEN inen_over_' + pair + '_' + chh)

    return get_kopplungs_ME()


def get_reduced_width_set(ws, inen, bsorsc='b', outf='WIDTHS_RED'):
    ind = 5 if bsorsc == 'b' else 6
    line = [
        int(1 - int(en))
        for en in [ll for ll in open(inen)][ind].split()[:len(ws)]
    ]
    masked_widths = np.ma.array(ws, mask=line)
    mma = []
    msa = ''
    for ww in masked_widths:
        if ww:
            mma.append(ww)
            msa += ww + '\n'
    with open(outf, 'w') as outfile:
        outfile.write(msa)
    return mma


def wid_gen(add=10, w0=w120, ths=[1e-5, 8e2, 0.1], sca=1.):
    tmp = []
    # rescale but consider upper and lower bounds
    for ww in w0:
        if ((float(ww) * sca < ths[1]) & (float(ww) * sca > ths[0])):
            tmp.append(float(float(ww) * sca))
        else:
            tmp.append(float(ww))
    tmp.sort()
    w0 = tmp
    #
    # add widths
    n = 0
    addo = 0
    while n < add:
        rn = random.randint(0, len(w0) - 1)
        rf = random.uniform(0.1, 2.)
        addo = float(w0[rn]) * rf
        tmp = np.append(np.array(w0), addo)
        tmp.sort()
        dif = min(abs(np.diff(tmp) / tmp[:-1]))
        #
        if ((addo > ths[0]) & (addo < ths[1]) & (dif > ths[2])):
            w0.append(addo)
            w0.sort()
            n = n + 1
    #
    w0.reverse()
    w0 = ['%12.6f' % float(float(ww)) for ww in w0]
    return w0


def repl_line(fn, lnr, rstr):
    s = ''
    fil = [line for line in open(fn)]
    for n in range(0, len(fil)):
        if n == lnr:
            s += rstr
        else:
            s += fil[n]
    with open(fn, 'w') as outfile:
        outfile.write(s)


def prep_pot_files_pdp(lam, wiC, baC, wir2, bar2, ls, ten, ps2):
    s = ''
    s += '  1  1  1  1  1  1  1  1  1\n'
    # pdp:       c p2 r2 LS  T Tp
    s += '  0\n%3d  0%3d%3d%3d  0  0\n' % (int((wiC != 0) | (baC != 0)),
                                           int((wir2 != 0) | (bar2 != 0)),
                                           int(ls != 0), int(ten != 0))
    # central LO Cs and Ct and LOp p*p' C_1-4
    if int((wiC != 0) | (baC != 0)):
        s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f\n' % (1.0,
                                                        float(lam)**2 / 4.0,
                                                        wiC, 0.0, baC)
    # r**2
    if int((wir2 != 0) | (bar2 != 0)):
        s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f\n' % (1.0,
                                                        float(lam)**2 / 4.0,
                                                        wir2, 0.0, bar2)
    # SPIN-BAHN
    if int(ls != 0):
        s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f\n' % (1.0,
                                                        float(lam)**2 / 4.0,
                                                        ls, 0.0, 0.0)
    # TENSOR
    if int(ten != 0):
        s += '%-20.4f%-20.6f%-20.4f%-20.4f%-20.4f\n' % (1.0,
                                                        float(lam)**2 / 4.0,
                                                        ten, 0.0, 0.0)
    with open(ps2, 'w') as outfile:
        outfile.write(s)
    return


def prep_pot_file_3N(lam3, ps3='', d10=0.0):
    s = ''
    s += '  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1\n  1  1\n'
    # pure central, no (iso)spin dependence
    s += '%-20.4f%-20.4f%-20.4f%-20.4f\n' % (d10, float(lam3)**2 / 4.0,
                                             float(lam3)**2 / 4.0,
                                             float(lam3)**2 / 4.0)
    # central, (s_j s_k)(t_j t_k) to project, set INEN factors +/- 4
    s += '%-20.4f%-20.4f%-20.4f%-20.4f' % (d10, float(lam3)**2 / 4.0,
                                           float(lam3)**2 / 4.0,
                                           float(lam3)**2 / 4.0)

    with open(ps3, 'w') as outfile:
        outfile.write(s)

    return


def parse_ev_coeffs(mult=0, outf='COEFF'):
    os.system('cp ' + 'OUTPUT ' + 'tmp')
    out = [line2 for line2 in open('OUTPUT')]
    #for n in range(1,len(out)):
    #    if(out[n].strip()=="EIGENWERTE DES HAMILTONOPERATORS"):
    #        print(float(out[n+3].split()[0]))
    coef = ''
    coeffp = []
    coeff_mult = []
    bvc = 0
    for line in range(0, len(out) - 1):
        if re.search('ENTWICKLUNG DES  1 TEN EIGENVEKTORS', out[line]):
            for bvl in range(line + 2, len(out)):
                if ((out[bvl][:3] == ' KO') | (out[bvl][:3] == '\n')):
                    bvc = out[bvl - 1].strip().split('/')[-1].split(')')[0]
                    break
                coeffp += [
                    float(coo.split('/')[0])
                    for coo in out[bvl].strip().split(')')[:-1]
                ]
                coef += out[bvl]
            break
    s = ''
    for n in range(len(coeffp)):
        if mult:
            for m in range(len(coeffp) - n):
                if m == 0:
                    s += '%18.10g' % (coeffp[n] * coeffp[n + m]) + '\n'
                # for identical fragments, c1*c2|BV1>|BV2> appears twice and can be summed up => faktor 2
                # see coef_mul_id.exe
                else:
                    s += '%18.10g' % (coeffp[n] * coeffp[n + m] * 2) + '\n'
        else:
            s += '%E' % (coeffp[n]) + '\n'
            #s += '%18.10g' % (coeffp[n]) + '\n'
    ss = s.replace('e', 'E')
    if bvc == 0:
        print("No coefficients found in OUTPUT")
    with open(outf, 'w') as outfile:
        outfile.write(ss)
    return


def read_phase(phaout='PHAOUT', ch=[1, 1], meth=1, th_shift=''):
    lines = [line for line in open(phaout)]

    th = {'': 0.0}
    phase = []
    phc = []
    ech = [0]

    for ln in range(0, len(lines)):
        if (lines[ln].split()[2] != lines[ln].split()[3]):
            th[lines[ln].split()[2] + '-' + lines[ln].split()[3]] = abs(
                float(lines[ln].split()[1]) - float(lines[ln].split()[0]))
    ths = th[th_shift]
    for ln in range(0, len(lines)):
        if ((int(lines[ln].split()[2]) == ch[0]) &
            (int(lines[ln].split()[3]) == ch[1]) &
            (int(lines[ln].split()[11]) == meth)):
            # energy tot -- energy th -- phase
            phase.append([
                float(lines[ln].split()[0]),
                float(lines[ln].split()[1]) + ths,
                float(lines[ln].split()[10])
            ])

    return phase


def write_phases(ph_array, filename='tmp.dat', append=0, comment=''):

    outs = ''

    if append == 1:
        oldfile = [line for line in open(filename)]
        for n in range(len(oldfile)):
            if oldfile[n].strip()[0] != '#':
                outs += oldfile[n].strip() + ' %12.8f' % float(
                    ph_array[n - 1][2]) + '\n'
            else:
                outs += oldfile[n]

    elif append < 0:
        oldfile = [line for line in open(filename)][:append]
        for n in range(len(oldfile)):
            if oldfile[n].strip()[0] != '#':
                for entry in oldfile[n].strip().split()[:append]:
                    outs += ' %12.8f' % float(entry)
                outs += ' %12.8f' % float(ph_array[n - 1][2]) + '\n'
            else:
                outs += oldfile[n]

    elif append == 0:
        outs = '#% -10s  %12s %12s' % ('E_tot', 'E_tot-Eth', 'Phase(s)\n')
        for line in range(len(ph_array)):
            outs += '%12.8f %12.8f %12.8f' % (float(ph_array[line][0]),
                                              float(ph_array[line][1]),
                                              float(ph_array[line][2]))
            outs += '\n'

    if comment != '': outs += comment + '\n'

    with open(filename, 'w') as outfile:
        outfile.write(outs)
    return


def plot_phases(phase_file,
                xlab='$E_{cm}\;\;\;\;[MeV]$',
                ylab='$\delta({}^2n-n)\;\;\;\;[Deg]$',
                legend_entry=''):

    pltcols = {
        '6.0': 'lightgray',
        '8.0': 'gray',
        '10.0': 'darkgray',
        '12.0': 'black'
    }
    lab = legend_entry

    en_ph_1_to_N = [line for line in open(phase_file) if line[0] != '#'][2:]

    nbr_of_phases = len(en_ph_1_to_N[0].split()) - 2

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel(r'%s' % xlab, fontsize=14)
    ax1.set_ylabel(r'%s' % ylab, fontsize=14)

    for n in range(2, 2 + nbr_of_phases):

        curcol = pltcols[[l for l in open(phase_file)
                          ][-nbr_of_phases + n - 2].split('=')[1].split('fm')[
                              0][:-1]] if n != nbr_of_phases + 1 else 'red'

        ax1.plot(
            [float(en.split()[0]) for en in en_ph_1_to_N],
            [float(ph.split()[n]) for ph in en_ph_1_to_N],
            color=curcol)

    plt.title(r'%s' % legend_entry, fontsize=16)

    fig.savefig(phase_file[:-3] + 'pdf')
    #plt.show()


def identicalt_stru(dir, spli):
    out = [line for line in open(dir + 'OUTPUT')]
    for n in range(1, len(out)):
        if (out[n].strip(
        ) == "ENTWICKLUNG DES  1 TEN EIGENVEKTORS,AUSGEDRUECKT DURCH NORMIERTE BASISVEKTOREN"
            ):
            sp = [t.strip() for t in out[n + 2:n + 2 + spli]]
            sp2 = [t.strip() for t in out[n + 2 + spli:n + 2 + 2 * spli]]
    le = 0
    for tt in sp:
        le += len(tt.split()) - 1
    le2 = 0
    for tt in sp2:
        le2 += len(tt.split()) - 1
    return le, le2


def determine_struct(inqua='INQUA_N'):
    """ 
    the number of basis vectors for given Spin- and Spatial coupling
    schemes is deduced from INQUA_N, where the width tuples associated
    with the scheme are put in a decadent ordered;
    'count #BV until the first width in the tuple is larger!'
    """

    lines_inqua = [line for line in open(inqua)]
    lines_inqua = lines_inqua[2:]

    stru = []

    bv_in_scheme = 0
    block_head = 0
    wr1_oldblock = 1e12

    while block_head < len(lines_inqua):

        bvinz = int(lines_inqua[block_head][:4])
        rel = int(lines_inqua[block_head + 1][4:7])
        nl = int(rel / 6)
        if rel % 6 != 0:
            nl += 1

        wr1_newblock = float(lines_inqua[block_head + 2].strip().split()[0])
        if wr1_newblock > wr1_oldblock:
            stru.append(bv_in_scheme)
            bv_in_scheme = 0

        #print(wr1_oldblock, wr1_newblock, block_head)

        bv_in_scheme += bvinz

        wr1_oldblock = float(
            lines_inqua[block_head + 1 + bvinz].strip().split()[0])

        if bvinz < 7:
            block_head = block_head + 2 + bvinz + nl + 2 * bvinz
        else:
            block_head = block_head + 2 + bvinz + nl + 3 * bvinz

    stru.append(bv_in_scheme)

    return [np.sum(stru[:n]) for n in range(1, len(stru) + 1)]
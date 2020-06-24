import os, sys
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.optimize import fmin

from LEC_fit_setup import *
from rrgm_functions import *
from three_particle_functions import *

j3 = 1 / 2

#6: (s1 s2)0 s3)1/2 , l1 l2 = '01' , Sc , (u)physical_NBR
#1: (s1 s2)1 s3)1/2 , l1 l2 = '10' , Sc , (u)physical_NBR
#2: (s1 s2)1 s3)3/2 , l1 l2 = '10' , Sc , (u)physical_NBR

basis_6135_t = [
    ['t_no6', '00', [1 / 2], 'p_1s0_00_05_05'],
    ['t_no1', '00', [1 / 2], 'p_2_10_05_05'],
    ['t_no3', '110', [1 / 2], 'p_3_00_05_05'],
    ['t_no3', '111', [1 / 2], 'p_3_00_05_05'],
    ['t_no5', '111', [1 / 2], 'p_3_00_05_05'],
    ['t_no5', '112', [1 / 2], 'p_3_00_05_05'],
]

basis_61352_t = [
    ['t_no6', '00', [1 / 2], 'p_1s0_00_05_05'],
    ['t_no1', '00', [1 / 2], 'p_2_10_05_05'],
    ['t_no3', '110', [1 / 2], 'p_3_00_05_05'],
    ['t_no3', '111', [1 / 2], 'p_3_00_05_05'],
    ['t_no5', '111', [1 / 2], 'p_3_00_05_05'],
    ['t_no5', '112', [1 / 2], 'p_3_00_05_05'],
    ['t_no2', '20', [1 / 2], 'p_4_10_05_05'],
    ['t_no2', '02', [1 / 2], 'p_4_10_05_05'],
]

basis_12_n3 = [
    ['1', '10', [1 / 2], 'p_1'],
    ['2', '10', [1 / 2], 'p_1'],
]

basis_2_n3 = [
    ['n3_no21', '111', [1 / 2], 'p_1'],
    ['n3_no22', '112', [1 / 2], 'p_1'],
    #['n3_no2', '10', [1 / 2], 'p_1'],
]

basis_61_he = [
    ['he_no6', '00', [1 / 2], 'p_1'],
    ['he_no1', '00', [1 / 2], 'p_1'],
]
basis_61_t = [
    ['t_no1', '00', [1 / 2], 'p_1'],
    ['t_no6', '00', [1 / 2], 'p_1'],
]

basis_61_n3 = [
    ['n3_no1', '10', [1 / 2], 'p_1'],  # S=1/2
    ['n3_no6', '01', [1 / 2], 'p_1'],  # S=1/2
]

basis = basis_61_t

two_body_structs = [
    'np-1S0', 'np-3S1', 'np-1P1', 'np-3P0', 'np-3P1', 'np-3P2', 'nn-1S0',
    'nn-3P0'
]

# relate 3n channel to 2n fragments for 2-1 scattering input
dict_3to2 = {
    'he_no6': two_body_structs[0],
    'he_no2': two_body_structs[1],
    'he_no1': two_body_structs[1],
    'he_no3': two_body_structs[2],
    'he_no5': two_body_structs[3],
    't_no6': two_body_structs[0],
    't_no2': two_body_structs[1],
    't_no1': two_body_structs[1],
    't_no3': two_body_structs[3],
    't_no5': two_body_structs[3],
    'n3_no6': two_body_structs[6],
    'n3_no21': two_body_structs[7],
    'n3_no22': two_body_structs[7],
    'n3_no1': two_body_structs[7]
}
typ = ''
for bv in basis:
    typ += bv[0] + '-'
typ = typ[:-1] + '__%d_%dP' % (int(str(j3)[0]), int(str(j3)[2]))
if dbg: print('3N basis structure: ', typ)

parit = 1  # 0/1 = +/-
optLECs = {}

for num_lam in range(len(Lrange)):
    print('L = %2.2f' % (Lrange[num_lam]))
    la = ('%-4.2f' % Lrange[num_lam])[:4]
    n2path = home + '/kette_repo/sim_par/nucleus/2n/' + la + '/'
    n3path = home + '/kette_repo/sim_par/nucleus/3N/' + la + '/'

    uec = []
    try:
        for partition in two_body_structs:
            uec.append([line for line in open(n2path + 'COEFF_' + partition)])
    except:
        print('>>> 2-body fragment coefficients not read! <<<')

    if os.path.isdir(n3path) == False:
        os.system('mkdir ' + n3path)
    if os.path.isfile(n3path + 'INQUA_N') == True:
        os.system('rm ' + n3path + 'INQUA_N')

    os.chdir(n3path)

    tnii = 13 if tni else 10
    nop = 26 if tni else 9

    if tni:
        coeff3 = np.array([0, 0., 1.])
        pots3 = 'pot_nnn_%02d' % int(float(la))

        opt2bdyLEC = lec_list[la][0]
        try:
            d0 = lec_list_unitary_scatt[la][-1]
        except:
            xy = np.array([[float(l), lec_list[l][1]] for l in lec_list.keys()
                           if float(l) >= 0.2])
            x = xy[:, 0]
            y = xy[:, 1]
            z = np.polyfit(x, y, 12)
            p = np.poly1d(z)
            d0 = p(Lrange[num_lam])
            print('>>> interpolated LECs for this cutoff: d0 = %4.4f MeV' % d0)

        prep_pot_file_3N(lam3=la, ps3=pots3, d10=d0)

    pots = n2path + 'pot_nn_%02d' % int(float(la))

    #[0,1]->[a,b]
    scal_up = 1.3
    scal_do = 0.9
    csca_rel = np.random.random(len(basis)) * (scal_up - scal_do) + scal_do

    if dbg: print('Scaling factors (w_rel): ', csca_rel)

    w3rt = [
        wid_gen(add=4, ths=[1e-5, 2e2, 0.2], w0=w120, sca=csca_rel[i])
        for i in range(len(basis))
    ]

    # ECCE! width sets must be equal for basis sets which are contributing to the same JJ channel
    w3r = []
    rel_set = 0
    for n in range(len(basis)):
        try:
            rel_set += 1 if ((basis[n][3] != basis[n - 1][3]) &
                             (n != 0)) else 0
        except:
            continue
        # test same/different rel sets
        w3r.append(w3rt[0])

    frgm = []
    frgmlu = []
    frgmob = []

    frgm = dn_inqua_21(
        basis3=basis,
        dicti=dict_3to2,
        relw=w3r[n],
        fn_inq=n2path + 'INQUA_N_',
        fn_inen=n2path + 'INEN_',
        fn=pots)

    cf_count = 0
    bv_count = 0

    for bv in range(len(basis)):
        frgmob += len(frgm[dict_3to2[basis[bv][0]]]) * [basis[bv][0]]
        frgmlu += len(frgm[dict_3to2[basis[bv][0]]]) * [basis[bv][1]]

        basis[bv].append(frgm[dict_3to2[basis[bv][0]]])

        if basis[bv][3][0] == 'p':
            cfs = [
                line
                for line in open(n2path + 'COEFF_' + dict_3to2[basis[bv][0]])
            ]
        else:
            cfs = []
        basis[bv].append([
            cfs, [bv_count, bv_count + sum(frgm[dict_3to2[basis[bv][0]]])],
            [cf_count, cf_count + len(cfs)]
        ])
        cf_count += len(cfs)
        bv_count += sum(frgm[dict_3to2[basis[bv][0]]])

    if dbg: print(bv_count, '\n', frgmob, '\n', frgmlu)  #, '\n', basis)

    if 'QUA' in cal:
        n3_inlu(len(frgmlu), 7, 'INLUCN', frgmlu)
        os.system(BINpath + 'LUDW_EFT_new.exe')
        n3_inob(frgmob, 5)
        os.system(BINpath + 'KOBER_EFT_nn.exe')
        repl_line('INQUA_N', 1, pots + '\n')
        os.system(BINpath + 'QUAFL_' + mpii + '.exe')
        os.system('cp QUAOUT quaout_interaction')

        if tni:
            n3_inlu(len(frgmlu), 8, 'INLU', frgmlu)
            os.system(BINpath + 'DRLUD_EFT.exe')
            n3_inob(frgmob, 15)
            #os.system(BINpath + 'DROBER_PRO.exe')
            os.system(BINpath + 'DROBER_EFT.exe')
            repl_line('INQUA_N', 1, pots3 + '\n')
            #os.system(
            #    '/home_th/kirscher/kette_repo/source/seriell/eft_cib_pool/DRQUA_EFT.exe'
            #)
            os.system(BINpath + 'DRQUA_EFT.exe')
            os.system(
                'cp DRQUAOUT drquaout_interaction && cp OUTPUT output_drqua')

    for n in range(len(eps_space)):

        coeff = np.array(
            #  coul, cent,p^2,r^2,            LS,        TENSOR, TENSOR_p
            [1., pot_scale * eps_space[n], 0., 0., 0., 0., 0.])
        costr = ''
        for fac in coeff:
            costr += '%12.6f' % float(fac)
        if tni:
            costr += '\n'
            for fac in coeff3:
                costr += '%12.6f' % float(fac)
            costr += '\n\n'

        if 'bdg' in cal:

            n3_inen_bdg(
                bv_count,
                j3,
                rw=rw_bdg,
                co=coeff,
                fn='INEN_BDG',
                pari=parit,
                nzop=nop,
                tni=tnii)
            os.system('cp INEN_BDG INEN')
            repl_line('INEN', 2, '%s\n' % costr)

            os.system('cp quaout_interaction QUAOUT')
            os.system('cp drquaout_interaction DRQUAOUT')
            os.system(BINpath + 'DR2END_' + mpii + '.exe')

            if dbg:
                print(
                    'LS-scheme: B(3,eps=%2.2f) = %4.4f MeV [' %
                    (eps_space[n], get_h_ev()[0]),
                    get_h_ev(n=4),
                    ']')

            os.system('grep -A 8 -m 1 " TE EIGENVEKTOR LIEFERT" OUTPUT')
            rrgm_functions.parse_ev_coeffs()
            os.system('cp OUTPUT end_out_b && cp INEN inen_b')

        if 'reduce' in cal:
            os.system('cp quaout_interaction QUAOUT')
            os.system('cp drquaout_interaction DRQUAOUT')
            reduce_3n(
                ch=typ,
                maxc3=maxCoef,
                minc3=minCoef,
                size3=sizeFrag,
                ncycl=ncycl,
                maxd=maxDiff,
                ord=0,
                tnii=nop,
                delpredd=delPcyc)
            if dbg:
                print('-- reduced B(3,%s) = %4.4f MeV' % (typ, get_h_ev()[0]))

            os.system('cp OUTPUT end_out_b && cp INEN inen_b')

        if 'over' in cal:
            for Lov in over_space:
                for pair in ['singel', 'tripl']:
                    kplME = overlap(
                        bipa=BINpath, chh=typ, Lo=Lov, pair=pair, mpi=mpii)
                    print('< %s | (%s-contact,L=%4.4f) | %s > = %4.4e' %
                          (typ, pair[:4], Lov, typ, kplME))

        if 'scatt' in cal:

            anzcg = n3_inen_str(
                basis,
                dict_3to2,
                coeff,
                wr=rw_per_phys.astype(int),
                dma=rw_per_dist,
                jay=j3,
                anzch=chs_in_dr2,
                nzop=nop,
                tni=tnii)

            os.system('cp INEN_STR INEN')
            repl_line('INEN', 2, '%s\n' % costr)

            os.system(BINpath + 'DR2END_' + mpii + '.exe')

            b3jj = get_h_ev()[0]

            if dbg:
                print('Fragment energies: ', get_bind_en(n=4))
                print('JJ-scheme: B(3) = %4.4f MeV' % b3jj)

            n3_spole(
                nzen=150,
                e0=0.001,
                d0=0.03,
                eps=epsi,
                bet=beta,
                nzrw=100,
                frr=0.06,
                rhg=8.0,
                rhf=1.0,
                # suggested weight factors: L=0:0.5 ; L=1:0.3 ; L=2:<0.2
                pw=[adapt_weight for n in basis])

            os.system(BINpath + 'S-POLE_PdP.exe')

        if 'plot' in cal:

            phases = read_phase(
                phaout='PHAOUT', ch=[1, 1], meth=1, th_shift='')

            params = '# L=%2.1f fm^-1  e=%4.3f  beta=%3.2f  dma=%d  size(1S0)=%d  adapt_w=%2.1f  %s B(3,jj)=%3.1f MeV' % (
                Lrange[num_lam], epsi, beta, len(rw_per_dist), size2Frag,
                adapt_weight, typ, b3jj)
            write_phases(
                phases, filename='../ph_2n-n.dat', append=0, comment=params)
            plot_phases(phase_file='../ph_2n-n.dat')

    if 'fit' in cal:

        def fitti(fac3, fitb, fix=0, blabla=0):
            repl_line('INEN', 3, '%12.6f%12.6f%12.6f\n' % (0.0, 0.0, fac3))
            os.system(BINpath + 'DR2END_' + mpii + '.exe')
            lines_output = [line for line in open('OUTPUT')]
            for lnr in range(0, len(lines_output)):
                if lines_output[lnr].find(
                        'EIGENWERTE DES HAMILTONOPERATORS') >= 0:
                    E_0 = lines_output[lnr + 3].split()[fix]
            return abs(float(E_0) + fitb)

        print('>>> commencing fit...')

        laold = ('%-4.2f' % (Lrange[num_lam] + 0.05))[:4]
        fac = 1.002
        ft_lo = fmin(fitti, fac, args=(energy2fit, 0, 1), disp=False)
        res_lo = fitti(ft_lo[0], 0.0, 0, 0)
        print('L = %2.2f:  %12.4f yields B(3)= %8.4f' %
              (Lrange[num_lam], d0 * ft_lo[0], res_lo))
        optLECs[la] = [opt2bdyLEC, d0 * ft_lo[0]]
        print(optLECs)

if 'fit' in cal:
    print(optLECs)
import os, sys

import numpy as np
from scipy.optimize import fmin

from bridge import *
from rrgm_functions import *
from two_particle_functions import *
from C0D1_lec_sets import *

w120 = [
    129.5665, 51.3467, 29.47287, 13.42339, 8.2144556, 4.447413, 2.939,
    1.6901745, 1.185236, 0.84300, 0.50011, 0.257369, 0.13852, 0.071429,
    0.038519, 0.018573, 0.0097261, 0.00561943, 0.002765, 0.00101
]
w12 = [
    12.95665, 5.13467, 2.947287, 1.342339, .82144556, .4447413, 2.939,
    1.6901745, 1.185236, 0.84300, 0.50011, 0.257369, 0.13852, 0.071429,
    0.038519, 0.018573, 0.0097261, 0.00561943, 0.002765, 0.00101
]

lec_list = {  #d0GS TNI-UIX  ZENTRAL NNN   PROJ           d0ES
    '0.05': [-1.551665, 0.324253617188],
    '0.10': [-2.241663, 0.2456829375],
    '0.16': [-3.25382381543, 0.0190507091372],
    '0.20': [-4.040227, -0.223247242969],
    '0.22': [-4.46682689578, -0.374771360196],
    '0.25': [-5.14850032428, -0.641645783435],
    '0.30': [-6.396510, -1.20682104512],
    '0.35': [-7.8083, -1.98064161217],
    '0.40': [-9.31007578154, -2.82776987047],
    '0.45': [-10.9761364604, -3.91309051448],
    '0.50': [-12.78163, -5.20893734689],
    '0.55': [-14.7258111587, -6.71792212184],
    '0.60': [-16.8090746843, -8.46827168904],
    '0.65': [-19.03174482, -10.4630055535],
    '0.70': [-21.39405, -12.7344455911],
    '0.75': [-23.8940370852, -15.2760761543],
    '0.80': [-26.53580, -18.1374520182],
    '0.90': [-32.23375, -24.8126425911],
    '0.95': [-35.29122, -28.6644794764],
    '1.00': [-38.48853, -32.9110143276],
    '1.05': [-41.82441, -37.5268116221],
    '1.10': [-45.29949, -42.5703870486],
    '1.20': [-52.66596, -53.9705109645],
    '1.50': [-78.11106, -100.476136019],
    '2.00': [-131.64598, -232.412644337],
    '3.00': [-280.45691, -854.107499493],
    '4.00': [-484.92093, -2495.36419052],
    '6.00': [-1060.7967, -16915.6302127],
}

n2path = home + '/kette_repo/sim_par/nucleus/2n/fit/'

if os.path.isdir(n2path) == False:
    os.system('mkdir ' + n2path)
if os.path.isfile(n2path + 'INQUA_N') == True:
    os.system('rm ' + n2path + 'INQUA_N')

os.chdir(n2path)

# mini- and maximal cutoff, increment -----------------------------------------
lmax = 30.0
inc = 0.05
lam = 0.10
print('L = %2.2f' % (lam))
la = ('%-4.2f' % lam)[:4]

exit()
# Gaussian basis --------------------------------------------------------------
basisdim0 = 15

laplace_loc, laplace_scale = 1., .4

exp0log, expmaxlog = -1, 1

winiLITlog = np.logspace(
    start=exp0log, stop=expmaxlog, num=basisdim0, endpoint=True, dtype=None)

wini0 = winiLITlog

addw = 8
addwt = 'middle'
scale0 = 1.
scale1 = 1.05
min_spacing = 0.2

# width set: deuteron
rw0 = wid_gen(
    add=addw, addtype=addwt, w0=wini0, ths=[1e-5, 2e2, 0.2], sca=scale0)
rw0 = sparsify(rw0, min_spacing)[::-1]

# width set: triton(d-N coordinate)
rw1 = wid_gen(
    add=addw, addtype=addwt, w0=w120, ths=[1e-5, 2e2, 0.2], sca=scale1)
rw1 = sparsify(rw1, min_spacing)[::-1]

print(rw0)
print(rw1)
nzf0 = int(np.ceil(len(rw0) / 20.0))

h2_inlu(anzo=7)
os.system(BINpath + 'LUDW_EFT_new.exe')
h2_inob(anzo=5)
os.system(BINpath + 'KOBER_EFT_nn.exe')

cloB = lec_list[la][0]
cloW = 0.

#                       wiC                  baC    wir2  bar2    ls  ten
prep_pot_files_pdp(lam, cloW, cloB, 0.0, 0.0, 0.0, 0.0, 'pot_nn')

coeff3 = np.array([0, 0., 1.])

d0 = lec_list[la][-1]

prep_pot_file_3N(lam3=la, ps3='pot_nnn', d10=d0)

h2_inqua(rw0, 'pot_nn')
os.system(BINpath + 'QUAFL_' + mpii + '.exe')

coeff = np.array([1., 1., 0., 0., 0., 0., 0.])
costr = ''
for fac in coeff:
    costr += '%12.6f' % float(fac)
h2_inen_bs(relw=rw0, costr=costr, j=1, ch=[2])
os.system(BINpath + 'DR2END_' + mpii + '.exe')


def fitti(fac2, fitb, fix=0, blabla=0):
    repl_line('INEN', 2, '%12.6f%12.6f%12.6f%12.6f%12.6f\n' % (1.0, fac2, 0.0,
                                                               0.0, 0.0))
    os.system(BINpath + 'DR2END_' + mpii + '.exe')
    lines_output = [line for line in open('OUTPUT')]
    for lnr in range(0, len(lines_output)):
        if lines_output[lnr].find('EIGENWERTE DES HAMILTONOPERATORS') >= 0:
            E_0 = lines_output[lnr + 3].split()[fix]
    return abs(float(E_0) + fitb)


deub = 1.0
lecold = cloB
leclist = {}

h2_inqua(rw0, 'pot_nn')
os.system('cp INQUA_N INQUA_N_rw0')
h2_inqua(rw1, 'pot_nn')
os.system('cp INQUA_N INQUA_N_rw1')

frgm1 = dn_inqua_neu(
    relw=w120,
    fn_inq=n2path + 'INQUA_N_rw0',
    fn_inen=n2path + 'INEN',
    fn=n2path + 'pot_nn')
frgm2 = dn_inqua_neu(
    relw=w120,
    fn_inq=n2path + 'INQUA_N_rw1',
    fn_inen=n2path + 'INEN',
    fn=n2path + 'pot_nn',
    appendd=True)

frgm = np.concatenate((np.array(frgm1), np.array(frgm2)))

n3_inlu(len(frgm), 7, 'INLUCN', frgm)
os.system(BINpath + 'LUDW_EFT_new.exe')
n3_inob(frgm1, frgm2, 5)
os.system(BINpath + 'KOBER_EFT_nn.exe')
repl_line('INQUA_N', 1, 'pot_nn' + '\n')
os.system(BINpath + 'QUAFL_' + mpii + '.exe')
n3_inlu(len(frgm), 8, 'INLU', frgm)
os.system(BINpath + 'DRLUD_EFT.exe')
n3_inob(frgm1, frgm2, 15)
os.system(BINpath + 'DROBER_EFT.exe')
repl_line('INQUA_N', 1, 'pot_nnn' + '\n')
os.system(BINpath + 'DRQUA_EFT.exe')
costr += '\n'
for fac in coeff3:
    costr += '%12.6f' % float(fac)
costr += '\n\n'

n3_inen_bdg(
    np.sum(frgm),
    1. / 2.,
    rw=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    costr=costr,
    fn='INEN',
    pari=1,
    nzop=26,
    tni=13)

os.system(BINpath + 'DR2END_' + mpii + '.exe')
exit()

print('>>> commencing fit...')
while lam < lmax:

    la = ('%-4.2f' % lam)[:4]
    cloB = lecold
    cloW = 0.

    prep_pot_files_pdp(lam, cloW, cloB, 0.0, 0.0, 0.0, 0.0, 'pot_nn')

    h2_inqua(rw0, 'pot_nn')
    os.system(BINpath + 'QUAFL_' + mpii + '.exe')
    fac = 1.001
    ft_lo = fmin(fitti, fac, args=(deub, 0, 1), disp=False)
    res_lo = fitti(ft_lo[0], 0.0, 0, 0)
    print('L = %2.2f:  %12.4f yields B(2)= %8.4f' % (lam, cloB * ft_lo[0],
                                                     res_lo))
    leclist[la] = cloB * ft_lo[0]
    lam += inc
    lecold = leclist[la]

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
lmax = 10.0
inc = 1.0
lam = 1.0
deub = 0.1
print('L = %2.2f' % (lam))
la = ('%-4.2f' % lam)[:4]

# Gaussian basis --------------------------------------------------------------
basisdim0 = 25

laplace_loc, laplace_scale = 1., .4

exp0log, expmaxlog = -1, 1

winiLITlog = np.logspace(
    start=exp0log, stop=expmaxlog, num=basisdim0, endpoint=True, dtype=None)

wini0 = winiLITlog

addw = 8
addwt = 'middle'
scale0 = .5

min_spacing = 0.02

# width set: deuteron
rw0 = wid_gen(add=addw, w0=wini0, ths=[1e-5, 2e2, 0.2], sca=scale0)

print(rw0)
nzf0 = int(np.ceil(len(rw0) / 20.0))

h2_inlu(anzo=7)
os.system(BINpath + 'LUDW_EFT_new.exe')
h2_inob(anzo=5)
os.system(BINpath + 'KOBER_EFT_nn.exe')
potnn = 'pot_nn'

cloW = float(lec_list['6.00'][0])

prep_pot_files([0.25 * lam**2], [cloW], [], [], [], potnn)

coeffnn = np.array(
    #  coul, cent,p^2,r^2,            LS,        TENSOR, TENSOR_p
    [1., 1., 0, 0, 0, 0, 0])
costrnn = ''
for fac in coeffnn:
    costrnn += '%12.6f' % float(fac)
rw = wid_gen(add=addw, w0=w120, ths=[1e-5, 3e2, 0.2], sca=scale0)
h2_inqua(rw, potnn)
os.system(BINpath + 'QUAFL_' + mpii + '.exe')
h2_inen_bs(relw=rw, costr=costrnn, j=1, ch=[2])
os.system(BINpath + 'DR2END_' + mpii + '.exe')
Bnn = get_h_ev()[0]


def fitti(fac2, fitb, fix=0, blabla=0):
    repl_line('INEN', 2, '%12.6f%12.6f%12.6f%12.6f%12.6f\n' % (1.0, fac2, 0.0,
                                                               0.0, 0.0))
    os.system(BINpath + 'DR2END_' + mpii + '.exe')
    lines_output = [line for line in open('OUTPUT')]
    for lnr in range(0, len(lines_output)):
        if lines_output[lnr].find('EIGENWERTE DES HAMILTONOPERATORS') >= 0:
            E_0 = lines_output[lnr + 3].split()[fix]
    return abs(float(E_0) + fitb)


lecold = cloW
leclist = {}
bdgs = []

print('>>> commencing fit...')
print('# l [fm^-2]     a_core [fm^-2]     C0 [MeV]')
while lam < lmax:

    la = ('%-4.2f' % lam)[:4]
    cloW = lecold

    prep_pot_files([0.25 * lam**2], [cloW], [], [], [], potnn)

    h2_inqua(rw, 'pot_nn')
    os.system(BINpath + 'QUAFL_' + mpii + '.exe')
    fac = 1.001
    ft_lo = fmin(fitti, fac, args=(deub, 0, 1), disp=False)
    res_lo = fitti(ft_lo[0], 0.0, 0, 0)

    print('{ %2.2f , %12.4f , %12.4f },' %
          (lam, 1.5 * np.abs(res_lo) * mn['137'] / MeVfm**2, cloW * ft_lo[0]))
    bdgs.append(res_lo)

    lam += inc
    lecold = cloW * ft_lo[0]
print('BDGS\n', bdgs)
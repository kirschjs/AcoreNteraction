import numpy as np

from parameters_and_constants import *
from rrgm_functions import *
from two_particle_functions import *
from pathlib import Path

home = str(Path.home())

BINpath = home + '/kette_repo/source/seriell/eft_sandkasten/'

tni = 1

mpii = '137'

# 'reduce' , 'bdg' , 'over' , 'QUA' , 'plot' , 'scatt'
cal = ['OBLU', 'bdg', 'QUA', 'plot', 'scatt']
cal = ['scatt']

cal = ['bdg', 'QUA', 'scatt', 'reduce']
cal = ['bdg', 'QUA', 'OBLU']
cal = ['bdg']

cal = ['fit']
cal = ['bdg', 'QUA']
cal = ['bdg', 'QUA', 'reduce']
cal = ['bdg', 'QUA', 'fit']

energy2fit = 3.0
lec_list = tmp
Lrange = np.array(list(tmp.keys())).astype(float)  #np.arange(0.1, 10., 0.5)

dbg = 0

pot_scale = 1.

anzs = 1
v_i = 1.0
v_e = 1.0
over_space = [15., 0.1, 0.0001]
eps_space = np.linspace(v_i, v_e, anzs)

r2w = 0.0
r2b = 0.0
ls = 0.56  #0.56 #0.16 #0.13
ten = 1.0

chs_in_dr2 = -4

anze = 22
epsi = 0.01
beta = 2.1
adapt_weight = 0.3
rw_per_phys = np.array(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
rw_per_dist = np.array(
    [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

rw_bdg = np.ones(20)

sizeFrag = 15
maxCoef = 1000
minCoef = 200
ncycl = 300
maxDiff = 0.01
delPcyc = 2
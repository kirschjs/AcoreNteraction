import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colorbar, colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection

from potcoeffs import *
from LECs_interpolation_constr import *

datadir = "/home/kirscher/kette_repo/AcoreNteraction/data/"

data15 = [line for line in open(datadir + 'Lc_oneonefive.dat')]
data3 = [line for line in open(datadir + 'Lc_onethree.dat')]
data4 = [line for line in open(datadir + 'Lc_onefour.dat')]
dataU = [line for line in open(datadir + 'Lc_unitthree.dat')]

f = plt.figure(figsize=(10, 8), dpi=95)
#f.suptitle(r'$local$', fontsize=14)

ax1 = f.add_subplot(111)

ax1.set_xlabel(r'$A$', fontsize=12)
ax1.set_ylabel(r'$\Lambda_c\;[fm^{-1}]$', fontsize=12)

xx15 = [int(dat.split()[0].strip()) for dat in data15]
yyoneonefive = [float(dat.split()[1].strip()) for dat in data15]

ax1.plot(xx, yyoneonefive, label=r'$B(3)=1.5~$MeV', c='g', ls='-', lw=2)

strFile = '/home/kirscher/kette_repo/AcoreNteraction/plots/' + 'lc_tmp.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
plt.savefig(strFile)
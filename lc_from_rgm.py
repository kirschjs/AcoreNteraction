import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import shutil
import imp
from matplotlib import cm, colorbar, colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection

import rad_interpolation

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

datadir = "/home/kirscher/kette_repo/AcoreNteraction/"

MeVfm = 197.3161329

lcheu = [[3, 0.4365749176763875], [4, 0.6772709881798779], [
    5, 0.9529882937207227
], [6, 1.44], [7, 1.728], [8, 2.0736], [9, 2.0736], [10, 2.48832], [
    11, 2.9859839999999997
], [12, 2.9859839999999997], [13, 3.5831807999999996], [
    14, 3.5831807999999996
], [15, 3.5831807999999996], [16, 4.299816959999999], [17, 5.159780351999999],
         [18, 5.159780351999999], [19, 5.159780351999999]]

data = [line for line in open(datadir + 'lc4_from_rgm_3-30.dat')][1:]
anz = len(data[0].strip().split()) - 1

f = plt.figure(figsize=(18, 12))
f.suptitle(r'local and $a=0.1\;fm^{-2}$', fontsize=14)

ax1 = f.add_subplot(111)

ax1.set_xlabel(r'A', fontsize=12)
ax1.set_ylabel(r'$\lambda_c\;[fm^{-1}]$', fontsize=12)

ax1.plot(
    [float(line.strip().split()[0]) for line in data],
    np.array([float(line.strip().split()[1]) for line in data]),
    label='$SVM data$',
    c=colors[0],
    ls=':')

ax1.plot(
    [aa[0] for aa in lcheu], [aa[1] for aa in lcheu],
    label='$RGM$',
    c=colors[1],
    ls='-')

#ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.legend(
    loc='center left', numpoints=1, fontsize=14, bbox_to_anchor=(1.0, .4))

strFile = 'lc_vgl.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
plt.savefig(strFile)

plt.show()
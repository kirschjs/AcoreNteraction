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

data_rad = [line for line in open(datadir + 'rad_core_3.dat')][1:]
anz = len(data_rad[0].strip().split()) - 1

f = plt.figure(figsize=(18, 12))
f.suptitle(r'', fontsize=14)

ax1 = f.add_subplot(111)

ax1.set_xlabel(r'$\Lambda$ [fm]', fontsize=12)
ax1.set_ylabel(r'$\vert\langle\vec{r}\rangle\vert^{1/2}$ [fm]', fontsize=12)

[
    ax1.plot(
        [float(line.strip().split()[0]) for line in data_rad],
        np.array([float(line.strip().split()[n + 1]) for line in data_rad]),
        label='$A_{core} = $%d' % (6 - n),
        c=colors[n],
        ls='--') for n in range(anz)
]

ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.legend(
    loc='center left', numpoints=1, fontsize=14, bbox_to_anchor=(1.0, .4))

strFile = 'rms_cores.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
plt.savefig(strFile)

plt.show()
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

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

datadir = "/home/kirscher/Dropbox/LO_p-wave_systems/LO_p-wave_results/detailed/3.0/"

MeVfm = 197.3161329

sysems = {}
sysemp = {}
for filename in os.listdir(datadir):
    if (filename[1] != 'a'):
        sysems[filename[:-4]] = [line for line in open(datadir + filename)][1:]
    else:
        sysemp[filename[:-4]] = [line for line in open(datadir + filename)][1:]

f = plt.figure(figsize=(18, 12))
f.suptitle(r'', fontsize=14)

ax1 = f.add_subplot(111)

ax1.set_xlabel(r'$\Lambda$ [fm]', fontsize=12)
ax1.set_ylabel(r'$\vert\langle\vec{r}\rangle\vert^{1/2}$ [fm]', fontsize=12)

for n in range(len(sysemp.keys())):
    dim = min(
        len([
            float(line.strip().split()[0])
            for line in sysems[list(sysemp.keys())[n][1:]]
        ]),
        len([
            float(line.strip().split()[0])
            for line in sysemp[list(sysemp.keys())[n]]
        ]))
    ax1.plot(
        [
            float(line.strip().split()[0])
            for line in sysems[list(sysemp.keys())[n][1:]]
        ][:dim],
        np.array([
            float(line.strip().split()[12])
            for line in sysemp[list(sysemp.keys())[n]]
        ][:dim]) / np.array([
            float(line.strip().split()[12])
            for line in sysems[list(sysemp.keys())[n][1:]]
        ][:dim]),
        label='%s' % (list(sysemp.keys())[n][1:]),
        c=colors[n],
        ls='--')

ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.legend(
    loc='center left', numpoints=1, fontsize=14, bbox_to_anchor=(1.0, .4))

strFile = 'rms_ratios.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
plt.savefig(strFile)

plt.show()
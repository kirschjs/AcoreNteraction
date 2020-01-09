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

mark = ['--v', '--<', '-->']
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

datadir = {
    '1.5':
    "/home/kirscher/Dropbox/LO_p-wave_systems/LO_p-wave_results/detailed/1.5/",
    '3.0':
    "/home/kirscher/Dropbox/LO_p-wave_systems/LO_p-wave_results/detailed/3.0/",
    '4.0':
    "/home/kirscher/Dropbox/LO_p-wave_systems/LO_p-wave_results/detailed/4.0/",
}

MeVfm = 197.3161329

f = plt.figure(figsize=(22, 16))
#f.suptitle(r'', fontsize=14)
ax2 = f.add_subplot(111)

ax2.set_xlabel(r'$A$', fontsize=44, labelpad=20)

ax2.set_ylabel(r'$\mathfrak{r}$ [fm]', fontsize=44, labelpad=20)

#for n in range(len(sysemp.keys())):
#    dim = min(
#        len([
#            float(line.strip().split()[0])
#            for line in sysems[list(sysemp.keys())[n][1:]]
#        ]),
#        len([
#            float(line.strip().split()[0])
#            for line in sysemp[list(sysemp.keys())[n]]
#        ]))
#    ax1.plot(
#        [
#            float(line.strip().split()[0])
#            for line in sysems[list(sysemp.keys())[n][1:]]
#        ][:dim],
#        np.array([
#            float(line.strip().split()[12])
#            for line in sysemp[list(sysemp.keys())[n]]
#        ][:dim]),
#        label='%s' % (list(sysemp.keys())[n][1:]))

dataset = 0
for b3 in datadir.keys():
    sysems = {}
    sysemp = {}

    for filename in os.listdir(datadir[b3]):
        if (filename[1] != 'a'):
            sysems[filename[:-4] + b3] = [
                line for line in open(datadir[b3] + filename)
            ][1:]
        else:
            sysemp[filename[:-4] + b3] = [
                line for line in open(datadir[b3] + filename)
            ][1:]

    Ru = np.array([
        sysems[sy][-1].strip().split()[12] for sy in list(sysems.keys())
    ]).astype(float)
    Au = np.array(
        [len(sy.split(b3)[0]) for sy in list(sysems.keys())]).astype(int)
    Asi = np.argsort(Au)

    As = np.take(Au, Asi)
    Rs = np.take(Ru, Asi)

    plt.xticks(As, fontsize=40, rotation=0)
    ax2.tick_params(axis='both', which='major', pad=15)

    plt.yticks(fontsize=40, rotation=0)

    ax2.plot(
        As,
        Rs,
        mark[dataset],
        markersize=22,
        label=r'$B(3)=%2.1f\;$ MeV' % float(b3))
    dataset += 1

#ax1.set_ylim([0, 5])
ax2.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

plt.legend(loc='best', numpoints=1, fontsize=40)

strFile = 'rms_radii.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
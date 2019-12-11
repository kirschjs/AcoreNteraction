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

datadir = {
    '3.0':
    "/home/kirscher/Dropbox/LO_p-wave_systems/LO_p-wave_results/detailed/3.0/",
    #'4.0':
    #"/home/kirscher/Dropbox/LO_p-wave_systems/LO_p-wave_results/detailed/4.0/",
    #'1.5':
    #"/home/kirscher/Dropbox/LO_p-wave_systems/LO_p-wave_results/detailed/1.5/",
}

MeVfm = 197.3161329

sysems = {}
sysemp = {}

for b3 in datadir.keys():
    print(b3)
    for filename in os.listdir(datadir[b3]):
        if (filename[1] != 'a'):
            sysems[filename[:-4] + b3] = [
                line for line in open(datadir[b3] + filename)
            ][1:]
        else:
            sysemp[filename[:-4] + b3] = [
                line for line in open(datadir[b3] + filename)
            ][1:]

f = plt.figure(figsize=(18, 12))
f.suptitle(r'', fontsize=14)

ax1 = f.add_subplot(211)

ax1.set_xlabel(r'$\Lambda$ [fm]', fontsize=12)
ax1.set_ylabel(r'$\vert\langle\vec{r}\rangle\vert^{1/2}$ [fm]', fontsize=12)

ax2 = f.add_subplot(212)

ax2.set_xlabel(r'$\Lambda$ [fm]', fontsize=12)
ax2.set_ylabel(r'$\vert\langle\vec{r}\rangle\vert^{1/2}$ [fm]', fontsize=12)

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
        ][:dim]),
        label='%s' % (list(sysemp.keys())[n][1:]))
    ax2.plot(
        [
            float(line.strip().split()[0])
            for line in sysems[list(sysemp.keys())[n][1:]]
        ][:dim],
        np.array([
            float(line.strip().split()[12])
            for line in sysemp[list(sysemp.keys())[n]]
        ][:dim]) / np.array([
            float(line.strip().split()[12])
            for line in sysemp[list(sysemp.keys())[1]]
        ][:dim]),
        label='%s' % (list(sysemp.keys())[n][1:]))

#ax1.set_ylim([0, 5])
ax1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
plt.legend(
    loc='center left', numpoints=1, fontsize=14, bbox_to_anchor=(1.0, .4))

strFile = 'rms_radii.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
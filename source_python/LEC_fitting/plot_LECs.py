import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colorbar, colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection

from parameters_and_constants import *

f = plt.figure(figsize=(13, 6))
f.suptitle(r'$m_\pi=137\,$MeV', fontsize=14)

ax1 = f.add_subplot(121)
ax2 = f.add_subplot(122)

xx = [float(ll) for ll in list(lec_list_one_three['137'].keys())]
yy3 = [float(cc[0]) for cc in list(lec_list_one_three['137'].values())]
yy4 = [float(cc[0]) for cc in list(lec_list_one_four['137'].values())]
xx15 = [float(ll) for ll in list(lec_list_one_onefive['137'].keys())]
yy15 = [float(cc[0]) for cc in list(lec_list_one_onefive['137'].values())]
ax1.set_xlabel(r'$\Lambda \;\;\;[fm^{-1}]$', fontsize=12)
ax1.set_ylabel(r'$2-body\;\;LEC\;\;\;[MeV]$', fontsize=12)
ax1.plot(xx15, yy15, 'r', label=r'$B(3)=1.5\,$MeV', alpha=0.4)
ax1.plot(xx, yy3, 'r--', label=r'$B(3)=3\,$MeV')
ax1.plot(xx, yy4, 'r', label=r'$B(3)=4\,$MeV', alpha=0.4)
ax1.legend(loc=0, fontsize=14)

ax2.set_yscale('log')
yy3 = [-float(cc[1]) for cc in list(lec_list_one_three['137'].values())]
yy4 = [-float(cc[1]) for cc in list(lec_list_one_four['137'].values())]
yy15 = [-float(cc[1]) for cc in list(lec_list_one_onefive['137'].values())]
ax2.set_xlabel(r'$\Lambda \;\;\;[fm^{-1}]$', fontsize=12)
ax2.set_ylabel(r'$3-body\;\;LEC\;\;\;[MeV]$', fontsize=12)
ax2.plot(xx15, yy15, 'm:', label=r'$B(3)=1.5\,$MeV', alpha=1)
ax2.plot(xx, yy3, 'm--', label=r'$B(3)=3\,$MeV')
ax2.plot(xx, yy4, 'm', label=r'$B(3)=4\,$MeV', alpha=1)
ax2.legend(loc=0, fontsize=14)

strFile = 'LECs.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colorbar, colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection

from potcoeffs import *
from LECs_interpolation_constr import *

AA = 5
acore = 0.4

Lrange = np.array(list(lec_list_oneMEVopt.keys())).astype(float)

f = plt.figure(figsize=(10, 11))
f.suptitle(r'$A = %d\;\;\;\;\;\;a_{core}=%4.4f$' % (AA, acore), fontsize=14)

et1 = [
    eta1([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) for l in list(lec_list_oneMEVopt.keys())
]
et2 = [
    eta2([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) for l in list(lec_list_oneMEVopt.keys())
]
et3 = [
    eta3([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) for l in list(lec_list_oneMEVopt.keys())
]
zet1 = [
    zeta1([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) for l in list(lec_list_oneMEVopt.keys())
]
zet2 = [
    zeta2([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) for l in list(lec_list_oneMEVopt.keys())
]
zet3 = [
    zeta3([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) for l in list(lec_list_oneMEVopt.keys())
]
zet4 = [
    zeta4([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) for l in list(lec_list_oneMEVopt.keys())
]

summaz = np.array(zet1) + np.array(zet2) + np.array(zet3) + np.array(zet4)
summae = np.array(et1) + np.array(et2) + np.array(et3)

ax1 = f.add_subplot(311)
ax2 = f.add_subplot(312)
ax3 = f.add_subplot(313)

ax1.set_xlabel(r'$\Lambda \;\;\;[fm^{-1}]$', fontsize=12)
ax1.set_ylabel(r'$\eta \;\;\;[MeV]$', fontsize=12)
ax1.plot(Lrange, et1, 'r--', label=r'$\eta_1$')
ax1.plot(Lrange, et2, 'k--', label=r'$\eta_2$')
ax1.plot(Lrange, et3, 'b--', label=r'$\eta_3$')
ax1.legend(loc=0, fontsize=14)
ax2.plot(Lrange, zet1, 'm--', label=r'$\zeta_1$')
ax2.plot(Lrange, zet2, 'r--', label=r'$\zeta_2$')
ax2.plot(Lrange, zet3, 'k--', label=r'$\zeta_3$')
ax2.plot(Lrange, zet4, 'b--', label=r'$\zeta_4$')

ax2.legend(loc=0, fontsize=14)
ax2.set_xlabel(r'$\Lambda \;\;\;[fm^{-1}]$', fontsize=12)
ax2.set_ylabel(r'$\zeta \;\;\;[MeV]$', fontsize=12)
ax3.plot(Lrange, summae, 'k-', label=r'$\sum\eta_i$')
ax3.plot(Lrange, summaz, 'b-', label=r'$\sum\zeta_i$')
ax3.plot(Lrange, summae - summaz, 'r--', label=r'$\sum(\eta_i-\zeta_i)$')
ax3.legend(loc=0, fontsize=14)
ax3.set_xlabel(r'$\Lambda \;\;\;[fm^{-1}]$', fontsize=12)
ax3.set_ylabel(r'$\sum \;\;\;[MeV]$', fontsize=12)
#plt.text(0, 225, r'$\eta_1$ and $\zeta_2$ 2-body', {
#    'color': 'r',
#    'fontsize': 20
#})
#plt.text(5, 225.0, r'$\eta_{2,3}$ and $\zeta_{3,4}$ 3-body', {
#    'color': 'b',
#    'fontsize': 20
#})

strFile = 'pottmp.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
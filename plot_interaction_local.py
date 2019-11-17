import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm, colorbar, colors
from matplotlib.ticker import FormatStrFormatter
from matplotlib.collections import LineCollection
from scipy.special import eval_genlaguerre, iv, spherical_jn

from potcoeffs import *
from LECs_interpolation_constr import *

AA = 6
acore = .5
rpSet = [0.01, 0.1, 1., 10.]

Lrange = np.array(list(lec_list_oneMEVopt.keys()))
Rrange = np.linspace(0, 5, 100)
Lset = Lrange[[
    0, 2, 3, 5,
    int(0.25 * len(Lrange)),
    int(0.5 * len(Lrange)),
    int(0.75 * len(Lrange)), -1
]]

f = plt.figure(figsize=(10, 10))
f.suptitle(
    r'$A = %d\;\;\;\;\;\;a_{core}=%4.4f\;\;\;\;\;\;R^\prime=%3.3f\ldots%3.3f$'
    % (AA, acore, rpSet[0], rpSet[-1]),
    fontsize=14)


def vloc1(r, l, acore, AA):

    return eta1([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) * np.exp(-kappa1([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) * r**2)


def vloc2(r, l, acore, AA):

    return eta2([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) * np.exp(-kappa2([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) * r**2)


def vloc3(r, l, acore, AA):

    return eta3([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) * np.exp(-kappa3([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) * r**2)


def vnlocSUM(rr, rl, l, acore, AA):
    Lrel = 1
    return np.real(-(1j**Lrel) * ((zeta2([
        acore, AA,
        float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
    ]) * np.nan_to_num(
        spherical_jn(Lrel, 1j * bet2([
            acore, AA,
            float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
        ]) * rr * rl),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0) * np.nan_to_num(
            np.exp(-alf2([
                acore, AA,
                float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
            ]) * rr**2 - gam2([
                acore, AA,
                float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
            ]) * rl**2),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )) + (zeta3([
            acore, AA,
            float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
        ]) * np.nan_to_num(
            spherical_jn(Lrel, 1j * bet3([
                acore, AA,
                float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
            ]) * rr * rl),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        ) * np.nan_to_num(
            np.exp(-alf3([
                acore, AA,
                float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
            ]) * rr**2 - gam3([
                acore, AA,
                float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
            ]) * rl**2),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        )) + (zeta4([
            acore, AA,
            float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
        ]) * np.nan_to_num(
            spherical_jn(Lrel, 1j * bet4([
                acore, AA,
                float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
            ]) * rr * rl),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0
        ) * np.nan_to_num(
            np.exp(-alf4([
                acore, AA,
                float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
            ]) * rr**2 - gam4([
                acore, AA,
                float(l), lec_list_oneMEVopt[l][0], lec_list_oneMEVopt[l][1]
            ]) * rl**2),
            nan=0.0,
            posinf=1.0,
            neginf=-1.0))))


print(vnlocSUM(1, 1, '0.10', acore, AA))

ax1 = f.add_subplot(211)
ax2 = f.add_subplot(212)
ax1.set_xlabel(r'$r \;\;\;[fm]$', fontsize=12)
ax1.set_ylabel(r'$V_{loc} \;\;\;[MeV]$', fontsize=12)
ax2.set_xlabel(r'$r \;\;\;[fm]$', fontsize=12)
ax2.set_ylabel(r'$V_{non-loc} \;\;\;[MeV]$', fontsize=12)

#[
#    ax1.plot(Rrange, [vloc1(rr, l, acore, AA) for rr in Rrange], label=r'')
#    for l in Lset
#]
#[
#    ax1.plot(Rrange, [vloc2(rr, l, acore, AA) for rr in Rrange], label=r'')
#    for l in Lset
#]
#[
#    ax1.plot(Rrange, [vloc3(rr, l, acore, AA) for rr in Rrange], label=r'')
#    for l in Lset
#]
[
    ax2.plot(
        Rrange, [vnlocSUM(r, rp, Lset[-1], acore, AA) for r in Rrange],
        label=r'$\sum_{i=2}^4 V^{\Lambda=%3.2f}_{non-loc,i}(r,r^\prime=%3.2f)$'
        % (float(Lset[-1]), rp)) for rp in rpSet
]
[
    ax1.plot(
        Rrange, [
            vloc1(rr, l, acore, AA) + vloc2(rr, l, acore, AA) +
            vloc3(rr, l, acore, AA) for rr in Rrange
        ],
        label=r'$\sum V_i(\Lambda=%3.3f)$' % float(l)) for l in Lset
]
ax1.legend(loc='best', fontsize=10)
ax2.legend(loc='best', fontsize=10)
strFile = 'potloc.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
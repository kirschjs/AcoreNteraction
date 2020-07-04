import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, iv, spherical_jn

data = np.array([
    np.array(line.split()).astype(float)
    for line in open('../data/B2is03_B3is3_ab_ab_100_200.res')
    if line[0] != '#'
])

data2 = np.array([
    np.array(line.split()).astype(float)
    for line in open('../data/B2is03_B3is3_ab_ac_100_200.res')
    if line[0] != '#'
])

f = plt.figure(figsize=(10, 12))

ax1 = f.add_subplot(211)
ax1.set_xlabel(r'$\lambda$ [fm$^{-1}$]', fontsize=12)
ax1.set_ylabel(r'$a_{DD}(\lambda)$]', fontsize=12)
ax1.plot(data[:, 1], data[:, 5], 'g-', lw=1, label="ab-ab")
ax1.plot(data2[:, 1], data2[:, 5], 'db-', alpha=0.5, lw=2, label="ab-ac")
plt.legend(loc='best', numpoints=1, fontsize=12)

ax1 = f.add_subplot(223)
ax1.set_xlabel(r'$\lambda$ [fm$^{-1}$]', fontsize=12)
ax1.set_ylabel(r'$C(\lambda)$]', fontsize=12)
ax1.plot(data[:, 1], data[:, 3], 'r-', lw=2, label="")

ax1 = f.add_subplot(224)
ax1.set_xlabel(r'$\lambda$ [fm$^{-1}$]', fontsize=12)
ax1.set_ylabel(r'$D(\lambda)$]', fontsize=12)
ax1.plot(data[:, 1], data[:, 4], 'b-', lw=2, label="")

strFile = os.getcwd() + '/tmp.pdf'
if os.path.isfile(strFile):
    os.remove(strFile)
plt.savefig(strFile)
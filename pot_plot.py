import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, iv, spherical_jn

v2H, a2H, b2H, c2H = -0.3706, -0.5821, -0.441, -0.5821
v3H, a3H, b3H, c3H = -12.94, -1.1441, -1.565, -1.1441
v4H, a4H, b4H, c4H = -331.2, -3.1108, -5.498, -3.1108

v2RGM, a2RGM, b2RGM, c2RGM = -2.0648, -0.684089, -0.680, -0.642044
v3RGM, a3RGM, b3RGM, c3RGM = -0.1298, -0.684089, -0.680, -0.642044
v4RGM, a4RGM, b4RGM, c4RGM = -0.1304, -0.688185, -0.680, -0.644092


def Vvv(rr, rl, v, a, b, c, L):

    vv = 0.
    for n in range(len(v)):
        vv += (1j**L) * (v[n] * spherical_jn(L, 1j * b[n] * rr * rl) * np.exp(
            a[n] * rr**2 + c[n] * rl**2))

    return np.nan_to_num(vv.real)


Ll = 1
rmin = 0
rmax = 6
dimR = 100
rspace = np.linspace(rmin, rmax, dimR)
potRGM = [
    Vvv(r, r, [v2RGM, v3RGM, v4RGM], [a2RGM, a3RGM, a4RGM],
        [b2RGM, b3RGM, b4RGM], [c2RGM, c3RGM, c4RGM], Ll) for r in rspace
]

potHiy = [
    Vvv(r, r, [v2H, v3H, v4H], [a2H, a3H, a4H], [b2H, b3H, b4H],
        [c2H, c3H, c4H], Ll) for r in rspace
]

f = plt.figure(figsize=(10, 6))

ax1 = f.add_subplot(121)
ax1.set_xlabel(r'$r$ [fm]', fontsize=12)
ax1.set_ylabel(r'$V_L(r,r\')$]', fontsize=12)
ax1.plot(rspace, potRGM, 'r-', lw=2, label="V(RGM)")
plt.legend(loc='bottom right', numpoints=1, fontsize=12)

ax1 = f.add_subplot(122)
ax1.plot(rspace, potHiy, 'b-', lw=2, label="V(Hiy)")
plt.legend(loc='bottom right', numpoints=1, fontsize=12)

plt.show()
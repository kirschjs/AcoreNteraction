import numpy as np


def eta1(argv):
    return argv[3] * (argv[1] - 1) * 8 / (4 + ((-1 + argv[1]) * argv[2]**2) / (argv[0] * argv[1]))**1.5


def kappa1(argv):
    return (argv[0] * argv[1] * argv[2]**2) / (4 * argv[0] * argv[1] +
                                               (-1 + argv[1]) * argv[2]**2)


def eta2(argv):
    return argv[4] * (argv[1] - 1) * (argv[1] - 2) * 64 * argv[0]**3 * (argv[1] /
                              (16 * argv[0]**2 * argv[1] + 4 * argv[0] *
                               (-1 + 3 * argv[1]) * argv[2]**2 +
                               (-2 + argv[1]) * argv[2]**4))**1.5

def kappa2(argv):
    return (2 * argv[0] * argv[1] * argv[2]**2 * (
        2 * argv[0] + argv[2]**2)) / (16 * argv[0]**2 * argv[1] + 4 * argv[0] *
                                      (-1 + 3 * argv[1]) * argv[2]**2 +
                                      (-2 + argv[1]) * argv[2]**4)


def eta3(argv):
    return argv[4] * (argv[1] - 1) * (argv[1] - 2) * 64 / (((4 * argv[0] + argv[2]**2) * (4 * argv[0] * argv[1] +
                                                (-2 + argv[1]) * argv[2]**2)) /
                 (argv[0]**2 * argv[1]))**1.5


def kappa3(argv):
    return (2 * argv[0] * argv[1] * argv[2]**2) / (4 * argv[0] * argv[1] +
                                                   (-2 + argv[1]) * argv[2]**2)


def zeta1(argv):
    return (2 * np.sqrt(2)) / (((-1 + argv[1]) * (1 + argv[1])**2) /
                               (argv[0] * argv[1]**3))**1.5


def alf1(argv):
    return (argv[0] * (argv[1] + argv[1]**3)) / (2. * (-1 + argv[1]) *
                                          (1 + argv[1])**2)


def bet1(argv):
    return (2 * argv[0] * argv[1]**2) / ((-1 + argv[1]) * (1 + argv[1])**2)


def gam1(argv):
    return (argv[0] * (argv[1] + argv[1]**3)) / (2. * (-1 + argv[1]) *
                                          (1 + argv[1])**2)


def zeta2(argv):
    return 8 / (np.pi**1.5 * (((1 + argv[1])**2 *
                               (4 * argv[0] * (-1 + argv[1]) +
                                (-2 + argv[1]) * argv[2]**2)) /
                              (argv[0]**2 * argv[1]**3))**1.5)


def alf2(argv):
    return (argv[0] * argv[1] *
            (4 * argv[0] * (1 + argv[1]**2) +
             (2 + argv[1] + 3 * argv[1]**2) * argv[2]**2)) / (2. * (
                 1 + argv[1])**2 * (4 * argv[0] * (-1 + argv[1]) +
                                    (-2 + argv[1]) * argv[2]**2))


def bet2(argv):
    return (4 * argv[0] * argv[1]**2 *
            (2 * argv[0] + argv[2]**2)) / ((1 + argv[1])**2 *
                                           (4 * argv[0] * (-1 + argv[1]) +
                                            (-2 + argv[1]) * argv[2]**2))


def gam2(argv):
    return (argv[0] * argv[1] *
            (4 * argv[0] * (1 + argv[1]**2) +
             (2 - argv[1] + argv[1]**2) * argv[2]**2)) / (2. * (
                 1 + argv[1])**2 * (4 * argv[0] * (-1 + argv[1]) +
                                    (-2 + argv[1]) * argv[2]**2))


def zeta3(argv):
    return (64 *
            (argv[0] * argv[1])**4.5) / (np.pi**1.5 * (1 + argv[1])**3 *
                                         (16 * argv[0]**2 *
                                          (-1 + argv[1]) + 4 * argv[0] *
                                          (-4 + 3 * argv[1]) * argv[2]**2 +
                                          (-3 + argv[1]) * argv[2]**4)**1.5)


def alf3(argv):
    return (argv[0] * argv[1] *
            (16 * argv[0]**2 * (1 + argv[1]**2) + 4 * argv[0] *
             (4 + argv[1] + 5 * argv[1]**2) * argv[2]**2 +
             (3 + 2 * argv[1] + 5 * argv[1]**2) * argv[2]**4)) / (
                 2. * (1 + argv[1])**2 *
                 (16 * argv[0]**2 * (-1 + argv[1]) + 4 * argv[0] *
                  (-4 + 3 * argv[1]) * argv[2]**2 +
                  (-3 + argv[1]) * argv[2]**4))


def bet3(argv):
    return (2 * argv[0] * argv[1]**2 *
            (16 * argv[0]**2 + 16 * argv[0] * argv[2]**2 + 3 * argv[2]**4)) / (
                (1 + argv[1])**2 *
                (16 * argv[0]**2 * (-1 + argv[1]) + 4 * argv[0] *
                 (-4 + 3 * argv[1]) * argv[2]**2 +
                 (-3 + argv[1]) * argv[2]**4))


def gam3(argv):
    return (argv[0] * argv[1] *
            (16 * argv[0]**2 * (1 + argv[1]**2) + 4 * argv[0] *
             (4 - argv[1] + 3 * argv[1]**2) * argv[2]**2 +
             (3 - 2 * argv[1] + argv[1]**2) * argv[2]**4)) / (
                 2. * (1 + argv[1])**2 *
                 (16 * argv[0]**2 * (-1 + argv[1]) + 4 * argv[0] *
                  (-4 + 3 * argv[1]) * argv[2]**2 +
                  (-3 + argv[1]) * argv[2]**4))


def zeta4(argv):
    return 64 / (np.pi**1.5 * (((1 + argv[1])**2 * (4 * argv[0] + argv[2]**2) *
                                (4 * argv[0] * (-1 + argv[1]) +
                                 (-3 + argv[1]) * argv[2]**2)) /
                               (argv[0]**3 * argv[1]**3))**1.5)


def alf4(argv):
    return (argv[0] * argv[1] *
            (4 * argv[0] * (1 + argv[1]**2) +
             (3 + 2 * argv[1] + 5 * argv[1]**2) * argv[2]**2)) / (2. * (
                 1 + argv[1])**2 * (4 * argv[0] * (-1 + argv[1]) +
                                    (-3 + argv[1]) * argv[2]**2))


def bet4(argv):
    return (2 * argv[0] * argv[1]**2 *
            (4 * argv[0] + 3 * argv[2]**2)) / ((1 + argv[1])**2 *
                                               (4 * argv[0] * (-1 + argv[1]) +
                                                (-3 + argv[1]) * argv[2]**2))


def gam4(argv):
    return (argv[0] * argv[1] *
            (4 * argv[0] * (1 + argv[1]**2) +
             (3 - 2 * argv[1] + argv[1]**2) * argv[2]**2)) / (2. * (
                 1 + argv[1])**2 * (4 * argv[0] * (-1 + argv[1]) +
                                    (-3 + argv[1]) * argv[2]**2))

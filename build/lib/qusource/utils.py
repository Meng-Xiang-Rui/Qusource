import numpy as np


def loc(x):
    """
    turn a string of 0/1 into its decimal number
    :param x: string of 0/1
    :return: Decimal number
    """
    return int(x, 2)


def length(state):
    """
    the length of the atoms chain of state.
    :param state: the quantum state
    :return: the length
    """
    return int(np.log2(len(state)))


def site_bit(number, N, site):
    """
    0/1 bit of the i-th atom in the base of N atoms chain from left-side
    :param number: state base
    :param N: length of atoms chain
    :param site: index of the atom in the atoms chain
    :return: 0/1
    """
    return number >> (N-site) & 1


def index(number, N, sites):
    """
    the index of number in the subspace generated by qubits in sites
    :param number: number
    :param N: length of atoms chain
    :param sites: sites of qubits
    :type sites: array-like
    :rtype: int
    """
    n = len(sites)
    res = 0
    for i in range(n):
        res += site_bit(number, N, sites[n - 1 - i]) * 2 ** i
    return res


def int2bin(n, count=24):
    """
    trans a number to binary string
    :param n: number
    :type n: int
    :param count: total length
    :type count: int
    :rtype: binary string
    """
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def set_bit(number, site, N, bit):
    """
    Change the bit value of number at a single site.

    :param number: the number to be changed
    :param site: the index of the site starting from left
    :param bit: the bit value to be changed
    :returns: the changed number
    """
    if bit:
        return number | (1 << (N-site))
    else:
        return number & ~(1 << (N-site))

import numpy as np
import scipy.stats as stats
from scipy.stats import bernoulli
from . import unitary, utils
# from .circuit import Circuit
import random


def distribution(x):
    """
    non-selective Z-measurement distribution
    :param x: state
    :type x: np.array
    :return: np.array
    """
    x /= np.linalg.norm(x, 2)
    prob = np.zeros(x.size)
    for i in range(len(x)):
        prob[i] = np.abs(x[i]) ** 2
    return prob


def dis2state(x):
    """
    turn a distribution into positive and real quantum state
    :param x: distribution
    :type x: np.array
    :return: np.array
    """
    state = np.array([np.sqrt(x[i]) for i in range(len(x))])
    return state


# TODO 添加多bases测量
def sample(state, n=1, bases=None, error=0):
    """
    n times measurement for quantum state.
    :param bases: measurement bases
    :param state: quantum state
    :param n: size of sample
    :return: n times measurement and its distribution
    :param error: measurement errors
    """
    N = utils.length(state)
    if not bases:
        measure = stats.rv_discrete(
            values=(range(len(state)), distribution(state))
        ).rvs(size=n)
        if not error:
            return measure[:n]
        else:
            return measure[:n], sample_error(measure[:n], N, error)
    elif isinstance(bases, str):
        new_state = unitary.trans_base(bases, state)
        measure = stats.rv_discrete(
            values=(range(len(new_state)), distribution(new_state))
        ).rvs(size=n)
        if not error:
            return measure[:n]
        else:
            return measure[:n], sample_error(measure[:n], N, error)
    else:
        measure = []
        if error:
            measure_error = []
        for base in bases:
            new_state = unitary.trans_base(base, state)
            measure.append(
                stats.rv_discrete(
                    values=(range(len(new_state)), distribution(new_state))
                ).rvs(size=n)
            )
            if error:
                measure_error.append(sample_error(measure[-1]), N, error)
        if not error:
            return np.array(measure)
        else:
            return np.array(measure), np.array(measure_error)


def ensemble_sample(ensembles, error=0):
    """
    one measurement for each state of an ensemble
    :param error: measurement
    :param ensembles: measurement ensembles
    :return: measurement results
    """
    N = utils.length(ensembles[0])
    measurement = []
    for tmp in ensembles:
        measurement.append(sample(tmp))
    if not error:
        return np.array(measurement)
    else:
        return np.array(measurement), sample_error(measurement, N, error)


def sample_error(samples, N, error=0):
    """
    Simulate the measurement error, each bit has a probability of error to be
    flipped. (first order approximation be used)
    :param samples: measurement results
    :param N: length of atoms chain
    :param error: probability of flips
    :return: measurement results with errors
    """
    size = len(samples)
    flip = bernoulli.rvs(N * error, size=size)
    ker = [2 ** i for i in range(N)]
    count = 0
    for i in range(size):
        if flip[i]:
            count += 1
            flip[i] = random.choice(ker)
    print("errors({0}), error rate = {1}.".format(count, count / size))
    sample_new = np.array([samples[i] ^ flip[i] for i in range(size)])
    return sample_new


def sample_save(samples, N, path=None):
    """
    save samples at path in 0/1 series
    :param samples: measurement results
    :param N: length of atoms chain
    :param path: path to be saved
    :return: None
    """
    path = path if path else str(N) + ' qubits_measurement.txt'
    f = open(path, mode='w')
    for i in samples:
        tmp = utils.int2bin(i, N)
        for j in tmp:
            f.write(j + ' ')
        f.write('\n')
    f.close()
    print("saved at " + path)


def bases_save(bases, M, path1=None, path2=None):
    """
    save measurement bases of each measurement at path1
    save the set of measurement bases at path2
    :param bases: the measurement bases
    :param M: measurement times
    :param path1: the path to save each measurement base
    :param path2: the path to save measurement bases set
    :return: None
    """
    N = len(bases[0])
    path1 = path1 if path1 else str(N) + ' qubits_measurement_bases.txt'
    path2 = path2 if path2 else str(N) + ' qubits_measurement_bases_set.txt'
    f1 = open(path1, mode='w')
    f2 = open(path2, mode='w')
    for base in bases:
        for i in base:
            f2.write(i + ' ')
            f2.write('\n')
        for _ in range(M):
            for i in base:
                f1.write(i + ' ')
                f1.write('\n')
    f1.close()
    f2.close()
    print("bases saved at " + path1)
    print("bases set saved at " + path2)


def sample_dis(samples, N):
    """
    turn a bunch of samples into distribution
    :param samples: samples
    :param N: size of total qubits
    :return: np.array
    """
    n = len(samples)
    sam_dis = np.zeros(2 ** N)
    for i in samples:
        sam_dis[i] += 1
    sam_dis /= n
    return sam_dis


def density_dis(x):
    """
    extract the distribution from a density matrix
    :param x:
    :return:
    """
    n = len(x)
    dis = np.array([x[i][i] for i in range(n)])
    return dis.real


def kld(p, q):
    """
    K-L divergence between distribution p and q.
    If some elements of p or q equal to zero, just remove these terms.
    :param p: distribution
    :param q: distribution
    :return: real number
    """
    divergence = 0
    for i in range(len(p)):
        if p[i] and q[i]:
            divergence += p[i] * np.log(p[i] / q[i])
    return divergence


def kld_(p, q):
    """
    K-L divergence between distribution p and q.
    A little trick to handle the zero case. Turn the infinity into a great number.
    :param p: distribution
    :param q: distribution
    :return: real number
    """
    n = len(p)
    epsilon = 0.01 / n
    p = p + epsilon
    q = q + epsilon
    divergence = np.sum(p * np.log(p / q))
    return divergence

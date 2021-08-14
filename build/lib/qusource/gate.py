import abc

import numpy as np
from scipy.linalg import expm
from . import utils, unitary


def noise(x):
    mu, sigma = x, x / 25.76
    np.random.seed()
    return np.random.normal(mu, sigma, 1)[0]


def sparse_check(x):
    tmp = x.flatten(order='C')
    nonzero = 0
    for i in range(len(tmp)):
        if tmp[i] != 0:
            nonzero += 1
    return nonzero, nonzero / len(tmp)


def unitary_check(x, threshold=1e-7):
    distance = np.linalg.norm(np.dot(unitary.dagger(x), x) - np.eye(len(x)))
    if distance < threshold:
        return True
    else:
        print('not unitary, error = {}'.format(distance))


# TODO 添加自定义noise_func的功能，更自定义一点
class Gate(abc.ABC):
    def __init__(self, ideal, *paras):
        self.ideal = ideal
        self.paras = paras
        self.ideal_gate = self.generator_(*self.paras)
        self.gate_shape = (self.ideal_gate != 0).astype('int')

    @abc.abstractmethod
    def generator_(self, *paras):
        pass

    def generator(self, noise_func=noise):
        if self.ideal:
            return self.ideal_gate
        else:
            paras = [noise_func(x) for x in self.paras]
            return self.generator_(*paras)

    def sparse_check(self):
        mat = self.generator()
        return sparse_check(mat)

    def unitary_check(self):
        mat = self.generator()
        return unitary_check(mat)


class Swap(Gate):
    def generator_(self, U, J, t, Delta=0):
        H = np.array([[U, -np.sqrt(2) * J, 0, 0, 0, 0, 0, 0, 0, 0],
                      [-np.sqrt(2) * J, Delta, -np.sqrt(2) * J, 0, 0, 0, 0, 0, 0, 0],
                      [0, -np.sqrt(2) * J, U + 2 * Delta, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, Delta, 0, -J, -J, 0, 0, 0],
                      [0, 0, 0, 0, Delta, -J, -J, 0, 0, 0],
                      [0, 0, 0, -J, -J, U, 0, 0, 0, 0],
                      [0, 0, 0, -J, -J, 0, U + 2 * Delta, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, U, -np.sqrt(2) * J, 0],
                      [0, 0, 0, 0, 0, 0, 0, -np.sqrt(2) * J, Delta, -np.sqrt(2) * J],
                      [0, 0, 0, 0, 0, 0, 0, 0, -np.sqrt(2) * J, U + 2 * Delta]])
        evolution = expm(H * 2 * np.pi * t * -1j)
        index = [1, 3, 4, 8]
        swap_mat = unitary.projection(evolution, index)
        swap_mat /= 1j
        return swap_mat


# TODO 将sto改成更general的sto gate
class Sto(Gate):
    def generator_(self, t, Delta):
        """
        only for 01/10 base
        :param t:
        :param Delta:
        :return:
        """
        phase = np.array(
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 2 * Delta, 0],
             [0, 0, 0, 0]]
        )
        evolution = expm(phase * 2 * np.pi * t * -1j)
        return evolution


class Dephase(Gate):
    def generator_(self, phase):
        phase_mat = np.diag([1, np.exp(-1j * phase)])
        return phase_mat

    def generator(self):
        amp = self.paras[0]
        phase = (noise(amp) - amp) * 2 * np.pi
        return self.generator_(phase)


# gate_map['swap'] = Swap
# gate_map['sto'] = Sto
# gate_map['dephase'] = Dephase


############################################################################################################
############################################################################################################
############################################################################################################
# def swap(U, J, t, Delta=0):
#     H = np.array([[U, -np.sqrt(2) * J, 0, 0, 0, 0, 0, 0, 0, 0],
#                   [-np.sqrt(2) * J, Delta, -np.sqrt(2) * J, 0, 0, 0, 0, 0, 0, 0],
#                   [0, -np.sqrt(2) * J, U + 2 * Delta, 0, 0, 0, 0, 0, 0, 0],
#                   [0, 0, 0, Delta, 0, -J, -J, 0, 0, 0],
#                   [0, 0, 0, 0, Delta, -J, -J, 0, 0, 0],
#                   [0, 0, 0, -J, -J, U, 0, 0, 0, 0],
#                   [0, 0, 0, -J, -J, 0, U + 2 * Delta, 0, 0, 0],
#                   [0, 0, 0, 0, 0, 0, 0, U, -np.sqrt(2) * J, 0],
#                   [0, 0, 0, 0, 0, 0, 0, -np.sqrt(2) * J, Delta, -np.sqrt(2) * J],
#                   [0, 0, 0, 0, 0, 0, 0, 0, -np.sqrt(2) * J, U + 2 * Delta]])
#     evolution = expm(H * 2 * np.pi * t * -1j)
#     index = [1, 2, 4, 8]
#     swap_mat = unitary.projection(evolution, index)
#     swap_mat /= 1j
#     return swap_mat
#
#
# def sto(t, Delta):
#     """
#     only for 01/10 base
#     :param t:
#     :param Delta:45
#     :return:
#     """
#     phase = np.array(
#         [[0, 0, 0, 0],
#          [0, 0, 0, 0],
#          [0, 0, 2 * Delta, 0],
#          [0, 0, 0, 0]]
#     )
#     evolution = expm(phase * 2 * np.pi * t * -1j)
#     return evolution
#
#
# swap_shape = np.array(
#     [[1, 0, 0, 0],
#      [0, 1, 1, 0],
#      [0, 1, 1, 0],
#      [0, 0, 0, 1]]
# )
#
# sto_shape = np.diag([1 for _ in range(4)])
#
# gate_shape = dict()
# gate_shape['swap'] = swap_shape
# gate_shape['sto'] = sto_shape
#
#
# def swap_noise(U, J, t, Delta=0):
#     return swap(noise(U), noise(J), noise(t), noise(Delta))
#
#
# def sto_noise(t, Delta):
#     return sto(noise(t), noise(Delta))
#
#
# def NumberOf1(n):
#     count = 0
#     while n & 0xffffffff != 0:
#         count += 1
#         n = n & (n - 1)
#     return count
#
#
# def phase_shift(n):
#     phase = 0
#     for _ in range(n):
#         phase += noise(2 * np.pi)
#     return phase
#
#
# def dephase(n):
#     numbers = []
#     for i in range(2 ** n):
#         numbers.append(NumberOf1(i))
#     numbers = np.array(numbers)
#     dephase_mat = np.diag([np.exp(phase_shift(i) * -1j) for i in numbers])
#     return dephase_mat

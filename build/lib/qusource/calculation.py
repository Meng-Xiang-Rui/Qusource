import numpy as np
from . import utils


# 反向计算 快
def fast_dot(sites, gate, state, N, layer=None):
    """
    A fast way to calculate the unitary transformation acting on some sites.

    :param sites: the sites on where the unitary transformation act
    :type sites: array like
    :param gate: the local transformation
    :param state: init quantum state
    :type state: dictionary
    :param layer: a list of index of final state
    :type layer: list
    :return: if state is a dictionary, then return a dictionary, else return a np.array
    """
    n = len(sites)
    layer = layer if layer else range(2 ** N)
    index = [utils.index(i, N, sites) for i in layer]
    gate = gate.T
    trans_mat = gate[:, index]
    v = np.array(layer).reshape(1, len(layer)).repeat(2 ** n, 0)
    for base in range(2 ** n):
        for site in range(n):
            v[base, :] = utils.set_bit(
                v[base, :], sites[site], N, utils.site_bit(base, n, site + 1)
            )

    if isinstance(state, dict):
        v = v.astype('complex')
        for i in range(len(v)):
            for j in range(len(v[i])):
                v[i, j] = state.get(v[i, j], 0)

        trans_mat *= v
        res = trans_mat.sum(0)
        final_state = dict(zip(layer, res))
        return final_state
    else:
        v = state[v]
        trans_mat *= v
        res = trans_mat.sum(0)
        return res


# 前向计算0 慢
def fast_dot_(sites, gate, state, N, layer=None):
    """
    A fast way to calculate the unitary transformation acting on some sites.

    :param sites: the sites on where the unitary transformation act
    :type sites: array like
    :param gate: the local transformation
    :param state: init quantum state
    :type state: dictionary
    :param layer: a list of index of final state
    :type layer: list
    :return: if state is a dictionary, then return a dictionary, else return a np.array
    """
    n = len(sites)
    layer = layer if layer else range(2 ** N)
    index = [utils.index(i, N, sites) for i in layer]
    trans_mat = gate[:, index]
    v = np.array(layer).reshape(1, len(layer)).repeat(2 ** n, 0)
    for base in range(2 ** n):
        for site in range(n):
            v[base, :] = utils.set_bit(
                v[base, :], sites[site], N, utils.site_bit(base, n, site + 1)
            )

    if isinstance(state, dict):
        v0 = np.array([state.get(x, 0) for x in layer])
        trans_mat *= v0
        final_state = dict()
        for i in range(len(trans_mat)):
            for j in range(len(trans_mat[i])):
                final_state[v[i, j]] = final_state.setdefault(v[i, j], 0) + trans_mat[i, j]
        return final_state
    else:
        v = state[v]
        trans_mat *= v
        res = trans_mat.sum(0)
        return res


# 前向计算 快
def fast_dot_2(sites, gate, state, N, layer=None, relation=None):
    """
    A fast way to calculate the unitary transformation acting on some sites.

    :param sites: the sites on where the unitary transformation act
    :type sites: array like
    :param gate: the local transformation
    :param state: init quantum state
    :type state: dictionary
    :param layer: a list of index of final state
    :type layer: list
    :return: if state is a dictionary, then return a dictionary, else return a np.array
    """
    # n = len(sites)
    layer = layer if layer else range(2 ** N)
    final_state = dict()
    for node in layer:
        index = utils.index(node, N, sites)
        for v in relation[node]:
            index1 = utils.index(v, N, sites)
            final_state[v] = final_state.setdefault(v, 0) + state[node] * gate[index1, index]
    return final_state


def fast_dot_1(sites, gate, state, N, layer=None, relation=None):
    """
    A fast way to calculate the unitary transformation acting on some sites.

    :param sites: the sites on where the unitary transformation act
    :type sites: array like
    :param gate: the local transformation
    :param state: init quantum state
    :type state: dictionary
    :param layer: a list of index of final state
    :type layer: list
    :return: if state is a dictionary, then return a dictionary, else return a np.array
    """
    # n = len(sites)
    layer = layer if layer else range(2 ** N)
    final_state = dict()
    for node in layer:
        index = utils.index(node, N, sites)
        for v in relation[node]:
            index1 = utils.index(v, N, sites)
            final_state[v] = final_state.setdefault(v, 0) + state[node] * gate[index1, index]
    return final_state



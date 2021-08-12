import numpy as np
from . import calculation


def projection(x, index, index1=None):
    """
    projection of state or operator in subspace {index}
    :param x: state or operator
    :param index: bases of subspace
    :param index1: bases of another subspace
    :return: projected state or operator
    """
    if len(x.shape) < 2:
        n = len(x)
        proj = np.eye(n)[index]
        return np.dot(proj, x)
    else:
        n = len(x[0])
        proj = np.eye(n)[index]
        if index1:
            proj1 = np.eye(n)[index1]
            return np.dot(np.dot(proj, x), proj1.T)
        else:
            return np.dot(np.dot(proj, x), proj.T)


def normalize(state):
    """
    normalization of state
    :param state:
    :return: np.array
    """
    tmp = state/np.linalg.norm(state)
    return tmp


def dagger(x):
    """
    Hermitian conjugate state
    :param x: state
    :return: Hermitian conjugate state
    """
    return x.T.conj()


def ketbra(state1, state2):
    """
    out product of state1 and state2
    :type state1: np.array
    :type state2: np.array
    :return: complex number
    """
    state1 = normalize(state1)
    state2 = normalize(state2)
    return np.outer(state1.conj(), state2)


def braket(state1, state2):
    """
    inner product of state1 and state2
    :type state1: np.array
    :type state2: np.array
    :return: np.array
    """
    state1 = normalize(state1)
    state2 = normalize(state2)
    return np.dot(state1.conj(), state2)


def density(ensembles):
    """
    density matrix of a ensemble of quantum states
    :param ensembles: np.array
    :return: np.array
    """
    if len(ensembles.shape) < 2:
        return ketbra(ensembles)
    else:
        den_mat = ketbra(ensembles[0])
        for i in range(1, len(ensembles)):
            den_mat += ketbra(ensembles[i])
        den_mat /= len(ensembles)
        return den_mat


def fidelity(state1, state2):
    """
    fidelity between state1 and state2, only valid for pure state1
    :type state1: np.array
    :type state2: np.array
    :return: real number
    """
    if len(state1.shape) > 1:
        print("error: state1 must be a pure state.")
    state1 = normalize(state1)
    fid = 0
    if len(state2.shape)<2:
        state2 = normalize(state2)
        fid = np.abs(
            braket(state1, state2)
        )**2
    else:
        for i in range(len(state2)):
            state2[i] = normalize(state2[i])
            fid += np.abs(
                braket(state1, state2)
            )**2
        fid /= len(state2)
    return fid.real


def purity(den_mat):
    """
    purity of density matrix
    :type den_mat: np.array
    :return: real number
    """
    return (np.trace(
        np.dot(den_mat, den_mat)
    )).real


def trans_base(bases, state, new_way=False):
    """
    Base transition.
    :param new_way: whether use the fast way or not
    :param bases: bases of each site
    :type bases: string
    :param state: quantum state
    :return: transformed quantum state
    """
    Z2Z = np.eye(2)
    Z2X = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    Z2Y = 1 / np.sqrt(2) * np.array([[1, -1j], [1, 1j]])
    decode = {'Z': Z2Z, 'X': Z2X, 'Y': Z2Y, 'z': Z2Z, 'x': Z2X,'y': Z2Y}
    if not new_way:
        tmp_mat = decode[bases[0]]
        for i in range(1, len(bases)):
            tmp_mat = np.kron(tmp_mat, decode[bases[i]])
        return np.dot(tmp_mat, state)
    else:
        for i in range(len(bases)):
            if bases[i] not in ['z', 'Z']:
                state = calculation.fast_dot(i, decode[bases[i]], state)
        return state


# def subsystem(sites, x):
#     """
#     density matrix of subsystem consisted of sites
#     :param sites: region of interest
#     :param x: state or density matrix
#     :return: reduced density matrix
#     """













import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import bernoulli
import random

trans_mat = np.zeros((4, 10))
trans_mat[0, 1] = trans_mat[1, 3] = trans_mat[2, 4] = trans_mat[3, 8] = 1


def int2bin(n, count=24):
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def loc(x):
    """
    turn a string of 0/1 into its decimal number
    :param x: string of 0/1
    :return: Decimal number
    """
    return int(x, 2)


def dagger(x):
    return x.T.conj()


def state_init(N, site):
    """
    :param N: length of state
    :param site: bit string of the site
    :type site: str
    :return: state
    """
    init_state = np.zeros(2**N)
    init_state[loc(site)] = 1
    return init_state


def W_state(N, log = False):

    coe = 1/pow(N, 0.5)
    state = np.zeros(2**N, dtype = 'complex')
    for i in range(N):
        state[2**i] = coe
    if log:
        f = open('w_psi'+str(N)+'.txt', mode='w')
        for i in range(2**N):
            f.write(str(state[i].real)+' '+str(state[i].imag)+'\n')
        f.close()
    return state


def state_save(state, path=None):
    N = int(np.log2(len(state)))
    path = path if path else str(N)+' qubits_state'+'.txt'
    f = open(path, mode='w')
    for i in range(2 ** N):
        f.write(str(state[i].real) + ' ' + str(state[i].imag) + '\n')
    f.close()
    print(path)


def amp_save(state, path=None):
    N = int(np.log2(len(state)))
    path = path if path else str(N)+' qubits_state_amp'+'.txt'
    f = open(path, mode='w')
    for i in range(2 ** N):
        f.write(str(np.abs(state[i])) + ' ' + str(0.0000) + '\n')
    f.close()
    print(path)


def sparse_check(x):
    tmp = x.flatten(order='C')
    nonzero = 0
    for i in range(len(tmp)):
        if tmp[i] != 0:
            nonzero += 1
    return nonzero, nonzero/len(tmp)


def unitary_check(x):
    threshold = 1e-10
    distance = np.linalg.norm(np.dot(dagger(x),x)-np.eye(len(x)))
    if distance<threshold:
        return True
    else:
        print('not unitary, error = {}'.format(distance))


def set_bit_val(byte, index, N, val):
    """
    更改某个字节中某一位（Bit）的值

    :param byte: 准备更改的字节原值
    :param index: 待更改位的序号，从右向左0开始，0-7为一个完整字节的8个位
    :param val: 目标位预更改的值，0或1
    :returns: 返回更改后字节的值
    """
    if val:
        return byte | (1 << (N-index))
    else:
        return byte & ~(1 << (N-index))


def site(data, N, i):
    return data >> (N-i) & 1


def fastmul(m,n, gate, state):
    N = int(np.log2(len(state)))
    index = [2*site(i,N,m)+site(i,N,n) for i in range(2**N)]
    gate = gate.T
    tmat = gate[:, index]
    v = np.arange(2**N).reshape(1,2**N).repeat(4,0)
    for i in range(4):
        p = site(i, 2, 1)
        q = site(i, 2, 2)
        v[i, :] = set_bit_val(v[i, :], m, N, p)
        v[i, :] = set_bit_val(v[i, :], n, N, q)
    v = state[v]
    tmat *= v
    res = tmat.sum(0)
    return res





def swap(U, J, t, Delta=0):
    H = np.array([[U, -np.sqrt(2)*J, 0, 0, 0, 0, 0, 0, 0,0],
              [-np.sqrt(2)*J, Delta, -np.sqrt(2)*J, 0, 0, 0, 0, 0, 0,0],
              [0, -np.sqrt(2)*J, U+2*Delta, 0, 0, 0, 0, 0, 0,0],
              [0, 0, 0, Delta, 0, -J, -J, 0, 0, 0],
              [0, 0, 0, 0, Delta, -J, -J, 0, 0, 0],
              [0, 0, 0, -J, -J, U, 0, 0, 0, 0],
              [0, 0, 0, -J, -J, 0, U+2*Delta, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, U, -np.sqrt(2)*J, 0],
              [0, 0, 0, 0, 0, 0, 0, -np.sqrt(2)*J, Delta, -np.sqrt(2)*J],
              [0, 0, 0, 0, 0, 0, 0, 0, -np.sqrt(2)*J, U+2*Delta]])
    Evolution = expm(H * 2*np.pi*t*-1j)
    swap = np.dot(trans_mat, Evolution)
    swap = np.dot(swap, trans_mat.T)
    swap /= 1j
    return swap


def sto(t, Delta):
    """
    only for 01/10 base
    :param t:
    :param Delta:
    :return:
    """
    phase = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 2*Delta, 0],
                      [0, 0, 0, 0]])
    Evolution = expm(phase * 2*np.pi*t*-1j)
    return Evolution


def noise(x):
    mu, sigma = x, x/25.76
    np.random.seed()
    return np.random.normal(mu, sigma, 1)[0]


def swap_noise(U, J, t, Delta = 0):
    #     U, J, t, Delta = noise(U), noise(J), noise(t), Delta = noise(Delta)
    return swap(noise(U), noise(J), noise(t), noise(Delta))


def sto_noise(t, Delta):
    return sto(noise(t), noise(Delta))


def NumberOf1(n):
    count = 0
    while n&0xffffffff != 0:
        count += 1
        n = n & (n-1)
    return count


def phase_shift(n):
    phase = 0
    for _ in range(n):
        phase += noise(2*np.pi)
    return phase


def dephase(n):
    numbers = []
    for i in range(2 ** n):
        numbers.append(NumberOf1(i))
    numbers = np.array(numbers)
    dephase_mat = np.diag([np.exp(phase_shift(i)*-1j) for i in numbers])
    return dephase_mat


def density_mat(x):
    n = len(x)
    sam_shape = len(x.shape)
    if sam_shape == 1:
        dim = len(x)
        state = x.reshape(1, dim)
        state /= np.linalg.norm(state)
        den_mat = np.dot(dagger(state), state)
    else:
        dim = len(x[0])
        den_mat = np.zeros((dim, dim))
        for i in range(n):
            state = x[i].reshape(1, dim)
            state /= np.linalg.norm(state)
            if not i:
                den_mat = np.dot(dagger(state), state)
            else:
                den_mat += np.dot(dagger(state), state)
        den_mat /= n
    return den_mat


def fidelity_vec(x, y):
    return (np.dot(x.conj(), y)*np.dot(y.conj(), x)/np.linalg.norm(x)**2/np.linalg.norm(y)**2).real


def fidelity_essemble(x,y):
    n = len(y)
    fidelity = 0
    for i in range(n):
        fidelity += fidelity_vec(x, y[i])
    return fidelity/n


def purity(x):
    return (np.trace(np.dot(x, x))).real


def distribution(x):
    x /= np.linalg.norm(x, 2)
    prob = np.zeros(x.size)
    for i in range(len(x)):
        prob[i] = np.abs(x[i])**2
    return prob


def dis2state(x):
    state = np.array([np.sqrt(x[i]) for i in range(len(x))])
    return state


def sample(x, n=1):
    N = int(np.log2(len(x)))
    res = stats.rv_discrete(values=(range(len(distribution(x))), distribution(x))).rvs(size=n)
    if n == 1:
        return res[0]
    else:
        dis = sample_distribution(res, N)
        kl = KL(distribution(x), dis)
        print(kl)
        return res, dis


def sample_distribution(sample, N):
    n = len(sample)
    sample_dis = np.zeros(2**N)
    for i in sample:
        sample_dis[i] += 1
    sample_dis /= n
    return sample_dis


def essemble_distribution(x):
    n = len(x)
    dis = np.array([x[i][i] for i in range(n)])
    return dis.real


def KL(p, q):
    divergence = 0
    for i in range(len(p)):
        if p[i] and q[i]:
            divergence += p[i]*np.log(p[i]/q[i])
    return divergence


def KL_new(P,Q):
    N = len(P)
    epsilon = 0.01/N
    P = P + epsilon
    Q = Q + epsilon
    divergence = np.sum(P*np.log(P/Q))
    return divergence


def sample_plot(dis, N, M, KL=None):
    x = [int2bin(i,N) for i in range(len(dis))]
    plt.bar(x, dis)
    plt.ylim(0,1)
    for x, y in enumerate(dis):
        plt.text(x, y+0.02, '%s' %y, ha='center')
    if not KL:
        plt.title('{} qubits with {} measurements'.format(N, M))
    else:
        plt.title('{} qubits with {} measurements\n KL = {}'.format(N, M, KL))
    plt.ylabel('Probility')
    plt.show()


def trans_base(bases, x):
    Z2Z = np.eye(2)
    Z2X = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    Z2Y = 1 / np.sqrt(2) * np.array([[1, -1j], [1, 1j]])
    decode = {'Z': Z2Z, 'X': Z2X, 'Y': Z2Y, 'z': Z2Z, 'x': Z2X,'y': Z2Y}
    tmp_mat = decode[bases[0]]
    for i in range(1,len(bases)):
        tmp_mat = np.kron(tmp_mat, decode[bases[i]])
    return np.dot(tmp_mat, x)


def sample_bases(bases, state, M):
    N = int(np.log2(len(state)))
    f1 = open(str(N)+' qubits_measurement.txt', mode='w')
    f2 = open(str(N)+' qubits_measurement_bases.txt', mode='w')
    f3 = open(str(N)+' qubits_measurement_bases_set.txt', mode='w')
    for i in bases:
        measure = sample(trans_base(i, state), M)
        for j in measure:
            tmp = int2bin(j, N)
            for k in tmp:
                f1.write(k+' ')
            f1.write('\n')
            for k in i:
                f2.write(k+' ')
            f2.write('\n')
        for j in i:
            f3.write(j+' ')
        f3.write('\n')
    f1.close()
    f2.close()
    f3.close()


def Z_sample(state, M, error=0):
    N = int(np.log2(len(state)))
    f1 = open(str(N)+' qubits_measurement_z.txt', mode='w')
    measure = sample(state, M)
    dis = sample_distribution(measure, N)
    kl = KL(dis, distribution(state))
    print(kl)
    if error:
        measure = sample_error(measure, N, error)
        dis = sample_distribution(measure, N)
        kl = KL(dis, distribution(state))
        print(kl)
    for j in measure:
        tmp = int2bin(j, N)
        for k in tmp:
            f1.write(k+' ')
        f1.write('\n')
    f1.close()


def sample_error(samples, n, error):
    size = len(samples)
    flip = bernoulli.rvs(n * error, size=size)
    ker = [2 ** i for i in range(n)]
    count = 0
    for i in range(size):
        if flip[i]:
            count += 1
            flip[i] = random.choice(ker)
    print(count, count / size)
    sample_new = np.array([samples[i] ^ flip[i] for i in range(size)])
    return sample_new


def sample_save(samples, N, path=None):
    path = path if path else str(N)+' qubits_measurement_z.txt'
    f1 = open(path, mode='w')
    for j in samples:
        tmp = int2bin(j, N)
        for k in tmp:
            f1.write(k+' ')
        f1.write('\n')
    f1.close()
    print(path)
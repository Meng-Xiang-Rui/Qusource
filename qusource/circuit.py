import numpy as np
from . import utils, gate, measurement
from .calculation import fast_dot


class Circuit:
    def __init__(self, N, sites, gates, paras, ideal=True, noise_fuc=None):
        """
        :param N: length of atoms chain
        :type sites: list of list(turtle)
        :type gates: list of string
        :type paras: dictionary from string to list(turtle)
        :type ideal: True or False
        :param noise_fuc: the function to generate noised parameters
        """
        self._N = N
        self.sites = sites
        self.gates = gates
        self.ideal = ideal
        self.paras = paras
        self.noise_fuc = noise_fuc
        self.steps = len(sites)
        self.init_state = dict()
        self.layer = []
        self.forward = []
        self.backward = []
        self.final_state = self.init_state

    @property
    def N(self):
        return self._N

    @property
    def space_size(self):
        """
        size of Hilbert space
        :return: int
        """
        return 2 ** self.N

    def hilbert_space(self):
        """
        Hilbert space
        :return: np.array
        """
        return np.arange(self.space_size)

    def state_init(self, state):
        """
        initialize state
        :param state: bit string of the site or a state vector
        :type state: str/np.array
        :return: np.array
        """
        if isinstance(state, str):
            self.init_state[utils.loc(state)] = 1+0j
            self.final_state = self.dic2array(self.init_state)
            self.trees(state)
        else:
            self.init_state = state
            self.final_state = self.init_state

    def dic2array(self, x):
        """
        a tool to transform a quantum state stored in a dictionary to a np.array
        :param x: quantum state
        :type x: dictionary
        :return: the transformed quantum state
        :rtype: np.array
        """
        state = np.zeros(self.space_size, dtype='complex')
        for i in x:
            state[i] = x[i]
        return state

    def trees(self, state):
        """
        build forward calculation tree\backward calculation tree and the layers
        :param state: initial state
        :type state: str
        :return: None
        """
        layer_tmp = [utils.loc(state)]
        self.layer.append(layer_tmp)

        gates_shape = {}
        for op in self.paras:
            opc = gate.gate_map[op]
            paras = self.paras[op]
            gates_shape[op] = opc(self.ideal, *paras).gate_shape

        for step in range(self.steps):
            sites = self.sites[step]
            n = len(sites)

            mat_shape = gates_shape[self.gates[step]]
            layer_tmp = set()
            map_tmp_f = dict()
            map_tmp_b = dict()

            for node in self.layer[-1]:
                node_index = utils.index(node, self.N, sites)
                sons = set()
                for base in range(2 ** n):
                    if mat_shape[base, node_index]:
                        tmp = node
                        for site in range(n):
                            tmp = utils.set_bit(
                                tmp, sites[site], self.N, utils.site_bit(
                                    base, n, site+1)
                            )
                        sons.add(tmp)
                map_tmp_f[node] = sons

                layer_tmp = layer_tmp.union(sons)
                for son in sons:
                    if son in map_tmp_b:
                        map_tmp_b[son].add(node)
                    else:
                        parent = set()
                        parent.add(node)
                        map_tmp_b[son] = parent

            self.layer.append(list(layer_tmp))
            self.forward.append(map_tmp_f)
            self.backward.insert(0, map_tmp_b)

    def branch(self, x):
        """
        generate a calculation branch of element x
        :param x: the x-th base of quantum state
        :return: the branch
        :rtype: a list of list
        """
        node_tmp = set()
        node_tmp.add(x)
        bran = [list(node_tmp)]
        for i in range(self.steps):
            node_tmp = set()
            for j in bran[0]:
                node_tmp = node_tmp.union(self.backward[i][j])
            bran.insert(0, list(node_tmp))
        return bran

    def state_generation(self):
        """
        generate a quantum state
        :return: generated quantum state
        """
        gates = dict()
        for op in self.paras:
            opc = gate.gate_map[op]
            paras = self.paras[op]
            gates[op] = opc(self.ideal, *paras)
        tmp_state = self.init_state
        for step in range(self.steps):
            sites = self.sites[step]
            op = gates[self.gates[step]].generator()
            if isinstance(tmp_state, dict):
                tmp_state = fast_dot(sites, op, tmp_state, self.N, self.layer[step+1])
                self.final_state = self.dic2array(tmp_state)
            else:
                tmp_state = fast_dot(sites, op, tmp_state, self.N)
                self.final_state = tmp_state
        return tmp_state

    def element(self, x):
        """
        calculate the x-th element of the final state
        :param x: base
        :return: psi(x), psi is the final state
        """
        branch = self.branch(x)
        gates = dict()
        for op in self.paras:
            opc = gate.gate_map[op]
            paras = self.paras[op]
            gates[op] = opc(self.ideal, *paras)
        tmp_state = self.init_state
        for step in range(self.steps):
            sites = self.sites[step]
            op = gates[self.gates[step]].generator()
            tmp_state = fast_dot(sites, op, tmp_state, self.N, branch[step+1])
        return tmp_state[branch[-1][0]]

    def ensemble_generation(self, size):
        """
        generate a ensemble of quantum state
        :param size: size of ensemble
        :return: ensembles of quantum state
        """
        ensemble = []
        for _ in range(size):
            ensemble.append(self.state_generation())
        ensemble = np.array(ensemble)
        self.final_state = ensemble
        return ensemble

    def state_amp(self, state=None):
        """
        the positive and real state of target state
        :param state: target state
        :return: the positive and real state of target state
        """
        state = state if isinstance(state, np.ndarray) else self.final_state
        amp_state = np.array(
            [np.abs(state[i]) for i in range(len(state))],
            dtype='complex')
        return amp_state

    def show(self, state=None):
        state = state if isinstance(state, np.ndarray) else self.final_state
        for i in range(len(state)):
            if np.abs(state[i]) > 1e-10:
                print('|'+utils.int2bin(i, self.N)+'>',
                      round(np.abs(state[i]), 3),
                      str(round(np.angle(state[i], False)/np.pi, 3))+'pi',
                      round(state[i].real, 3)+1j*round(state[i].imag, 3)
                      )

    def state_save(self, state=None, path=None):
        """
        save quantum state at path
        :param state:
        :param path:
        :return: None
        """
        state = state if isinstance(state, np.ndarray) else self.final_state
        path = path if path else str(self.N) + ' qubits_state' + '.txt'
        f = open(path, mode='w')
        for i in range(self.space_size):
            f.write(str(state[i].real) + ' ' + str(state[i].imag) + '\n')
        f.close()
        print('state saved at ' + path)

    def amp_save(self, state, path=None):
        """
        sate the positive and real state of state at path
        :param state:
        :param path:
        :return: None
        """
        state = state if isinstance(state, np.ndarray) else self.final_state
        path = path if path else str(self.N) + ' qubits_state_amp' + '.txt'
        amp_state = self.state_amp(state)
        self.state_save(amp_state, path)

    def mc_sample(self, size, error=0):
        """
        Monte Carlo method to generate samples
        :param size: size of samples
        :param error: measurement error
        :return: measurement results, (measurement results with measurement error)
        """
        self.state_generation()
        cover_func = 2 * measurement.distribution(self.final_state)
        x_list = self.sample(n=int(2.5*size), state=self.final_state)
        y_list = np.random.uniform(size=int(2.5*size))          
        count = 0
        measure = []
        for i in range(len(x_list)):
            y0 = self.element(x_list[i])
            if cover_func[x_list[i]] * y_list[i] < np.abs(y0)**2:
                count += 1
                measure.append(x_list[i])
                if count == size:
                    break
        if not error:
            return np.array(measure)
        else:
            return np.array(measure), measurement.sample_error(
                np.array(measure), self.N, error
            )

    def sample(self, n=1, state=None, bases=None, error=0):
        """
        generate samples
        :param n: size of smaples
        :param state: quantum state
        :param bases: measurement bases
        :param error: measurement error
        :return: sample results
        """
        state = state if isinstance(state, np.ndarray) else self.final_state
        if len(state.shape) < 2:
            return measurement.sample(state, n, bases, error)
        else:
            return measurement.ensemble_sample(state, error)

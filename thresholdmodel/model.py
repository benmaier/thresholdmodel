import numpy as np
import networkx as nx

import random

_S = 0
_I = 1

class ThreshModel():
    """
    A simple simulation class that runs 
    a threshold-model activation process
    on a static network (potentially weighted and directed)
    in continuous time using Gillespie's 
    stochastic simulation algorithm.

    The temporal dimension is fixed by assuming
    that every node whose activation threshold
    has been exceeded by neighboring inputs
    is activated with constant and uniform
    rate :math:`\gamma = 1`.

    Parameters
    ==========
    G : networkx.Graph, networkx.DiGraph
        The network on which to simulate.
        Nodes must be integers in the range
        of ``[0, N-1]``.
    initially_activated: float, int, or list of ints
        Can be either of three things:

        1. float of value ``0 < initially_activated < 1``.
           In this case, ``initially_activated`` is
           interpreted to represent a fraction of nodes
           that will be randomly selected from the
           set of nodes and set to be activated.
        2. int of value ``1 <= initially_activated < N-1``.
           In this case, ``initially_activated`` nodes
           will be randomly sampled from the node set
           and set to be activated.
        3. list of ints. In this case, ``initially_activated``
           is interpreted to contain indices of nodes
           that will be activated initially.
    thresholds: float or iterable of floats
        Can be either of two things:

        1. float of value ``0 < thresholds <= 1``.
           In this case, every node will have the same
           activation threshold.
        2. iterable of values ``0 < thresholds <=1``.
           In this case, the function expectes a list,
           tuple, or array with length equal to the
           number of nodes. Each entry `m` of this list
           will be interpreted to be node `m`'s activation
           threshold.
    weight: str, default = None
        A string that represents the weight keyword of a link.
        If `None`, the network is assumed to be unweighted.

    Example
    =======

    >>> G = nx.fast_gnp_random_graph(1000,20/(1000-1))
    >>> model = TreshModel(G, 100, 0.1)
    >>> t, cascade_size = model.simulate()

    Attributes
    ==========
    G : nx.Graph or nx.DiGraph
        The network on which to simulate.
        Nodes must be integers in the range
        of ``[0, N-1]``.
    N : int
        The number of nodes in the network
    weight: str
        A string that represents the weight keyword of a link.
        If `None`, the network is assumed to be unweighted.
    in_deg : numpy.ndarray
        Contains the in-degree of every node.
    thresholds: numpy.ndarray
        Each entry `m` of this array
        represents node `m`'s activation
        threshold.
    initially_activated: numpy.ndarray
        Each entry of this array contains
        a node that will be activated initially.
    time: numpy.ndarray
        Contains every time point at which a node was
        activates (after ``simulation()`` was called).
        The temporal dimension is given by assuming
        that every node whose activation threshold
        has been exceeded by activation inputs
        is activated with constant and uniform
        rate :math:`\gamma = 1`.
    cascade_size: numpy.ndarray
        The relative size of the activation cascade
        at the corrsponding time value in ``time``
        (relative to the size of the node set).
        Only available after ``simulation()`` was called.
    activated_nodes: list
        A list of lists. 
        Each entry contains a list of integers representing
        the nodes that have been activated  
        at the corrsponding time value in ``time``.
        Each list entry will contain only a single node
        for every other time than the initial time.
        Only available after ``simulation()`` was called.
    """
    
    def __init__(self,
            G,
            initially_activated,
            thresholds,
            weight=None,
            ):

        self.G = G
        self.N = self.G.number_of_nodes()
        self.weight = weight

        assert(set(self.G.nodes()) == set(list(range(self.N))))

        if nx.is_directed(G):
            self.in_deg = np.array([self.G.in_degree(i,weight=weight) for i in range(self.N) ],dtype=float)
        else:
            self.in_deg = np.array([self.G.degree(i,weight=weight) for i in range(self.N) ],dtype=float)

        self.set_initially_activated(initially_activated)
        self.set_thresholds(thresholds)

    def set_thresholds(self,thresholds):
        """
        Set node activation thresholds.

        Parameters
        ==========
        thresholds: float or iterable of floats
            Can be either of two things:

            1. float of value ``0 < thresholds <= 1``.
               In this case, every node will have the same
               activation threshold.
            2. iterable of values ``0 < thresholds <=1``.
               In this case, the function expectes a list,
               tuple, or array with length equal to the
               number of nodes. Each entry `m` of this list
               will be interpreted to be node `m`'s activation
               threshold.
        """

        if not hasattr(thresholds,"__len__"):
            assert(thresholds > 0 and thresholds <= 1)
            thresholds = np.ones(self.N,dtype=float) * thresholds

        else:
            assert(len(thresholds) == N)
            assert(np.all(thresholds <= 1) and np.all(thresholds>=0))

        self.thresholds = np.array(thresholds,dtype=float) * self.in_deg


    def set_initially_activated(self,initially_activated):
        """
        Set the process's initial activation state.

        Parameters
        ==========
        initially_activated: float, int, or list of ints
            Can be either of three things:

            1. float of value ``0 < initially_activated < 1``.
               In this case, ``initially_activated`` is
               interpreted to represent a fraction of nodes
               that will be randomly selected from the
               set of nodes and set to be activated.
            2. int of value ``1 <= initially_activated < N-1``.
               In this case, ``initially_activated`` nodes
               will be randomly sampled from the node set
               and set to be activated.
            3. list of ints. In this case, ``initially_activated``
               is interpreted to contain indices of nodes
               that will be activated initially.
        """

        if not hasattr(initially_activated,"__len__"):
            if initially_activated < 1 and initially_activated > 0:
                initially_activated = int(self.N*initially_activated)
            elif initially_activated < 0:
                raise ValueError("`initially_activated` must be a (i) float >0 and <1, (ii) positive integer, or (iii) a list of nodes")
            elif initially_activated > self.N-1:
                raise ValueError("if `initially_activated` is a positive integer, it has to meet < N-1.")

            initially_activated = np.random.choice(list(range(self.N)),size=initially_activated,replace=False)
        else:
            assert(len(initially_activated) <= self.N-1)

        self.initially_activated = np.array(initially_activated,dtype=int)


    def reset(self):
        """
        Reset the simulation.
        """

        self.node_status = np.zeros((self.N,),dtype=int)
        self.node_status[self.initially_activated] = _I

        self.nodes_that_will_flip = set()
        self.activation_influx = np.zeros((self.N,),dtype=float)

        nodes_with_influx = set()
        for a in self.initially_activated:
            for neigh in self.G.neighbors(a):
                if self.node_status[neigh] == _S:
                    if self.weight is None:
                        self.activation_influx[neigh] += 1
                    else:
                        self.activation_influx[neigh] += self.G.edges[a,neigh][self.weight]
                    nodes_with_influx.add(neigh)

        for i in nodes_with_influx:
            if self.activation_influx[i] >= self.thresholds[i]:
                self.nodes_that_will_flip.add(i)

        self.t = 0.0
        self.time = [self.t]
        self.A = len(self.initially_activated)
        self.cascade_size = [self.A]
        self.activated_nodes = [self.initially_activated.tolist()]

    
    def simulate(self,save_activated_nodes=False):
        """
        Simulate until all nodes that can be activated
        have been activated.

        Parameters
        ==========
        save_activated_nodes: bool, default = False
            If ``True``, write a list of activated nodes 
            to the class attribute ``activated_nodes``
            every time an activation event happens.
            Such a list will contain only a single node
            for every other time than the initial time.

        Returns
        =======
        time : numpy.ndarray
            Time points at which nodes were activated.
        cascade_size : numpy.ndarray
            The relative size of the activation cascade
            at the corrsponding time value in ``time``
            (relative to the size of the node set).
        """

        self.reset()

        while len(self.nodes_that_will_flip) > 0:

            mean_tau = 1.0/len(self.nodes_that_will_flip)
            tau = np.random.exponential(mean_tau)

            activated_node = random.sample(self.nodes_that_will_flip,1)[0]
            self.node_status[activated_node] = _I
            self.nodes_that_will_flip.remove(activated_node)

            for neigh in self.G.neighbors(activated_node):
                if self.node_status[neigh] == _S:
                    if self.weight is None:
                        self.activation_influx[neigh] += 1
                    else:
                        self.activation_influx[neigh] += self.G.edges[activated_node,neigh][self.weight]
                    if self.activation_influx[neigh] >= self.thresholds[neigh]:
                        self.nodes_that_will_flip.add(neigh)

            self.t += tau
            self.A += 1
            self.time.append(self.t)
            self.cascade_size.append(self.A)

            if save_activated_nodes:
                self.activated_nodes.append([activated_node])

        return np.array(self.time,dtype=float), np.array(self.cascade_size,dtype=float) / self.N

if __name__=="__main__":

    from sixdegrees.utils import to_networkx_graph
    from sixdegrees import small_world_network, modular_hierarchical_network
    import sixdegrees as sixd

    from bfmplot import pl

    import netwulf

    B = 10
    L = 3
    N = B**L
    k = 10
    mu = -0.8
    thresholds = 0.1
    initially_infected = np.arange(10)

    #G = to_networkx_graph(*modular_hierarchical_network(B,L,k,mu))
    G = to_networkx_graph(*sixd.random_geometric_kleinberg_network(N,k,mu))

    Thresh = ThreshModel(G,initially_infected,thresholds)
    t, a = Thresh.simulate()

    pl.plot(t,a)
    pl.ylim([0,1])
    pl.show()




import numpy as np
import networkx as nx

import random

_S = 0
_I = 1

class ThreshModel():
    
    def __init__(self,
            G,
            initially_activated,
            thresholds,
            weight=None,
            ):

        self.G = G
        self.N = self.G.number_of_nodes()
        self.weight = weight

        if nx.is_directed(G):
            self.in_deg = np.array([self.G.in_degree(i,weight=weight) for i in G.nodes()],dtype=float)
        else:
            self.in_deg = np.array([self.G.degree(i,weight=weight) for i in G.nodes()],dtype=float)

        self.set_initially_activated(initially_activated)
        self.set_thresholds(thresholds)

    def set_thresholds(self,thresholds):

        if not hasattr(thresholds,"__len__"):
            assert(thresholds > 0 and thresholds <= 1)
            thresholds = np.ones(self.N,dtype=float) * thresholds

        else:
            assert(len(thresholds) == N)
            assert(np.all(thresholds <= 1) and np.all(thresholds>=0))

        self.thresholds = np.array(thresholds,dtype=float) * self.in_deg


    def set_initially_activated(self,initially_activated):

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

        self.node_status = np.zeros((self.N,),dtype=int)
        self.node_status[self.initially_activated] = _I

        self.nodes_that_will_flip = set()
        self.activation_influx = np.zeros((self.N,),dtype=int)

        nodes_with_influx = set()
        for a in self.initially_activated:
            for neigh in self.G.neighbors(a):
                if self.node_status[neigh] == _S:
                    if not self.weight:
                        self.activation_influx[neigh] += 1
                    else:
                        self.activation_influx[neigh] += self.G.edges[a,neigh][weight]
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

        self.reset()

        while len(self.nodes_that_will_flip) > 0:

            mean_tau = 1.0/len(self.nodes_that_will_flip)
            tau = np.random.exponential(mean_tau)

            activated_node = random.sample(self.nodes_that_will_flip,1)[0]
            self.node_status[activated_node] = _I
            self.nodes_that_will_flip.remove(activated_node)

            for neigh in self.G.neighbors(activated_node):
                if self.node_status[neigh] == _S:
                    if not self.weight:
                        self.activation_influx[neigh] += 1
                    else:
                        self.activation_influx[neigh] += self.G.edges[activated_node,neigh][weight]
                    if self.activation_influx[neigh] >= self.thresholds[neigh]:
                        self.nodes_that_will_flip.add(neigh)

            self.t += tau
            self.A += 1
            self.time.append(self.t)
            self.cascade_size.append(self.A)

            if save_activated_nodes:
                self.activated_nodes.append([a])

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




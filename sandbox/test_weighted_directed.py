import numpy as np
import networkx as nx

from thresholdmodel import ThreshModel

N = 10

thresholds = 1.0
initially_infected = list(range(1,N))

G = nx.DiGraph()
G.add_edges_from([(i,0) for i in initially_infected])

Thresh = ThreshModel(G,initially_infected,thresholds)
t, a = Thresh.simulate(save_activated_nodes=True)

print(t)
print(a)
print(Thresh.activated_nodes)

G = nx.DiGraph(weight='flux')
G.add_edges_from([(i,0,{'flux':0.1}) for i in initially_infected])

Thresh = ThreshModel(G,initially_infected,thresholds,weight='flux')
t, a = Thresh.simulate(save_activated_nodes=True)

print(t)
print(a)
print(Thresh.activated_nodes)

G = nx.DiGraph(weight='flux')
thresholds = 0.9
G.add_edges_from([(i,0,{'flux':0.1}) for i in initially_infected])
initially_infected = list(range(1,N-2))
Thresh = ThreshModel(G,initially_infected,thresholds,weight='flux')
t, a = Thresh.simulate(save_activated_nodes=True)

print(t)
print(a)
print(Thresh.activated_nodes)

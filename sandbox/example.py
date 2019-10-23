import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from thresholdmodel import ThreshModel

N = 1000
k = 10

thresholds = 0.1
initially_infected = np.arange(100)

G = nx.fast_gnp_random_graph(N, k/(N-1.0))

Thresh = ThreshModel(G,initially_infected,thresholds)
t, a = Thresh.simulate()

plt.plot(t,a)
plt.ylim([0,1])
plt.xlabel('time $[1/\gamma]$')
plt.ylabel('cascade size ratio')
plt.gcf().savefig('cascade_trajectory.png')
plt.show()

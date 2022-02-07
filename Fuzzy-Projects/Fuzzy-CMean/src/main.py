import numpy as np
import matplotlib.pyplot as plt
from clustring import Clustering
from CONF import *

def data(path):
    return np.genfromtxt(path, delimiter=',')

def convert_to_color(U):
    color = []
    DIC = {0:'b', 1:'g', 5:'r', 3:'c', 4:'m', 2:'y', 6:'k', 7:'w'}
    for i in range(U.shape[1]):
        color.append(DIC[np.argmax(U[:, i])])
    return color


X = data("../data/data1.csv")

###  best C is 3
# Cs = []
# Js = []
# for c in CONDIDATES:
#     print(f"c = {c}")
#     clustring = Clustering(c, m, X)
#     Cs.append(c)
#     Js.append(clustring.run(iterations))

# plt.plot(Cs, Js)
# plt.show()

clustring = Clustering(C, m, X)
J, V, U = clustring.run(iterations), clustring.V, clustring.U
for i in range(X.shape[0]):
    plt.plot(X[i, 0], X[i, 1], '.', color=convert_to_color(U)[i])
plt.plot(V[:, 0], V[:, 1], 'or')
plt.show()

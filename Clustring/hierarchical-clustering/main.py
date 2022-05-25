import numpy as np
import matplotlib.pyplot as plt
import os

def loadbar(iter, total, prefix='', sufix='', decimals=1, length=100, fill='█', space=' '):
	percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iter / float(total)))
	fl = int(length * iter // total)
	bar = fill * fl + space * (length - fl)
	os.system("cls")
	print(f'{prefix} |{bar}| {percent}% {sufix}', end = '')
	if iter == total :
		print()


class Clustring:
    def __init__(self, data):
        self.DICTIONARY = {}
        self.init_DICTIONARY(data)
    
    
    def init_DICTIONARY(self, data):
        for i in range(data.shape[1]):
            self.DICTIONARY[i] = [data[:, i]]
    
    def merge_clusters(self, c1, c2):
        self.DICTIONARY[c1] = self.DICTIONARY[c1] + self.DICTIONARY[c2]
        del self.DICTIONARY[c2]
    
    def get_cluster_centers(self, c):
        cs = self.DICTIONARY[c]
        mean_ = cs[0]
        for i in range(1, len(cs)):
            mean_ = mean_ + cs[i]
        mean_ = mean_ / len(cs)
        return mean_
    
    def find_closest_clusters(self):
        In = []
        for i in self.DICTIONARY:
            In.append(i)
        x, y = In[0], In[1]
        min_ = np.linalg.norm(self.get_cluster_centers(x) - self.get_cluster_centers(y))
        for i in In:
            for j in In:
                if i != j:
                    if np.linalg.norm(self.get_cluster_centers(i) - self.get_cluster_centers(j)) < min_:
                        min_ = np.linalg.norm(self.get_cluster_centers(i) - self.get_cluster_centers(j))
                        x, y = i, j
        return x, y
    
    def run(self, k):
        s = 0
        l = len(self.DICTIONARY) - k - 1
        loadbar(0, l, prefix='Progress: ', sufix='Complete', length=l, space=' ')
        while len(self.DICTIONARY) > k:
            x, y = self.find_closest_clusters()
            self.merge_clusters(x, y)
            s += 1
            loadbar(s, l, prefix='Progress: ', sufix='Complete', length=l, space=' ')

        return self.DICTIONARY
    
    def create_clusters(self, k):
        self.run(k)
        DIC = {}
        j = 0
        for i in self.DICTIONARY:
            DIC[j] = self.DICTIONARY[i]
            j += 1
        return DIC
    
def c_xy(c):
    x = []
    y = []
    for i in c:
        x.append(i[0])
        y.append(i[1])
    return x, y


X = []
for i in range(400, 600, 2):
    y = np.random.randint(40, 60)
    X.append([i / 10, y])

for i in range(0, 200, 2):
    y = np.random.randint(0, 20)
    X.append([i / 10, y])


for i in range(800, 1000, 2):
    y = np.random.randint(0, 20)
    X.append([i / 10, y])


X = np.array(X).T

x = X[0]
y = X[1]
plt.scatter(x, y)
plt.show()

c = Clustring(X)
s = c.create_clusters(3)
C = ["ro", "go", "bo", "ko"]
for i in s:
    x, y = c_xy(s[i])
    plt.plot(x, y, C[i])
plt.show()

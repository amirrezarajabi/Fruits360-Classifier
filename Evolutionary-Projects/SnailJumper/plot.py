import numpy as np
import matplotlib.pyplot as plt

f = open("json.rj", 'r')
data = f.readlines()
means = []
maxs = []
mins = []
Is = []
x = 1
for d in data:
    d = d.replace("\n", "")
    d = d.split(",")
    Is.append(x)
    x += 1
    means.append(float(d[0]))
    maxs.append(float(d[1]))
    mins.append(float(d[2]))

plt.plot(Is, means, label="Mean")
plt.plot(Is, maxs, label="Max")
plt.plot(Is, mins, label="Min")
plt.legend()
plt.show()

erase = input("Do you want to erase the file? (y/n)")
if erase == "y":
    f = open("json.rj", 'w')
    f.write("")
    f.close()
else:
    pass
import matplotlib.pyplot as plt
import numpy as np


data = np.random.multivariate_normal([5, 5], [[1, 0], [0, 0.5]], 300)
plt.plot(*zip(*(data)), marker="8", ls='', markersize=8, markerfacecolor='w', markeredgecolor='b', markeredgewidth=2)
for i in data:
    if(i[0] - i[1] > 0):
        plt.text(i[0] - 0.01, i[1] - 0.01, "2", fontdict=None)
    else:
        plt.text(i[0] - 0.01, i[1] - 0.01, "1", fontdict=None)
plt.show()

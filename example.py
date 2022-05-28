import numpy as np
import matplotlib.pyplot as plt
from clustering import Cluster

x = np.random.random_sample((200, 2)) * 200
model = Cluster(5)
model.fit(x)
points = model.make_random(200/((1 + np.random.random_sample()) * 5), 4)
cntr = model.centroid()

plt.scatter(x[:, 0], x[:, 1])
plt.scatter(cntr[:, 0], cntr[:, 1])

for x in range(5):
    pts = points[x]
    plt.scatter(pts[:, 0], pts[:, 1])
plt.show()


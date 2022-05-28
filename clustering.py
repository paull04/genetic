from sklearn.cluster import KMeans
import numpy as np


class Cluster:
    def __init__(self, k):
        self.k = k
        self.k_means = KMeans(init="k-means++", n_clusters=k, n_init=12)
        self._centroid = None

    def fit(self, x):
        self.k_means.fit(x)
        self._centroid = self.k_means.cluster_centers_

    def centroid(self):
        self._centroid = np.array(self.k_means.cluster_centers_)
        return self._centroid

    def make_random(self, r, n):
        self.centroid()
        return np.array([
            [x + r * (2 * np.random.random_sample(2) - 1) for _ in range(n)]
            for x in self._centroid
        ])

"""
처음 아이디어는 클러스터를 만들고 랜덤하게 센트로이드에 변이를 주는 거였다.
변형된 센트로이드들에게 인덱스를 부여하고
인덱스들을 유전자로 한다.
지금은 센트로이드의 좌표를 처음부터 랜덤으로 잡고 이를 유전자로 하는 것으로 변경
"""
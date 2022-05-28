import numpy as np
import matplotlib.pyplot as plt
from eval_cluster import *


class Make:
    def __init__(self, candidate, n, max_num, r):
        self.can =  candidate#후보수
        self.nums = np.random.randint(max_num, size=(n, candidate)) #인구수, 지지하는 정당에 대해 몇 명이 지지하는 지에 대한 정보
        self.pos = np.random.random_sample((n, 2)) * r # 점들의 위치를 랜덤하게 결정
        self.total = np.sum(self.nums)
        self.votes = np.array([np.sum(self.nums[:, x]) for x in range(candidate)]) / self.total

    def __str__(self):
        s1 = str(self.can) + '\n'
        s2 = ''
        s3 = '\n' + str(self.pos)
        for w, x in enumerate(self.nums):
            s2 += f'{w}: ' + ' ,'.join([str(e) for e in x]) + '\n'

        return s1 + s2 + s3

    def cal(self, cntr):
        #cntr: centroid
        # 적합도 계산
        res = cal_cluster(cntr, self.pos, self.can, self.nums)
        res2 = eval_cluster(res, self.votes)
        res3 = is_alive(cntr, self.pos, self.nums)
        """
        첫번째 실험의 경우
        res = is_alive(cntr, self.pos, self.nums)
        return res
        """
        return res3, res2

    def scatter(self):
        plt.scatter(self.pos[:, 0], self.pos[:, 1])




if __name__ == '__main__':
    from clustering import Cluster
    p = Make(7, 300, 1000, 40)
    model = Cluster(10)
    model.fit(p.pos)
    cntr = model.centroid()
    res = cal_cluster(cntr, p.pos, p.can, p.nums)
    res2 = eval_cluster(res, p.votes)
    plt.scatter(p.pos[:, 0], p.pos[:, 1])
    plt.scatter(cntr[:, 0], cntr[:, 1])
    plt.show()
    print(res2)


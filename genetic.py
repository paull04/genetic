import numpy as np
from numpy.random import choice, random_sample, randint, shuffle, random
from make_data import Make
from eval_cluster import is_alive, eval_cluster


def arith(a, b):
    """
    p는 [0,1](균등 분포)사이값 하나를 랜덤으로 선택
    산술 연산은 다음과 같다.
    """
    p = random_sample()
    return a*p + b*(1-p)


def mutation(a, r):
    """
    랜덤으로 유전자 위치를 결정
    해당 위치에 랜덤한 값을 넣느다.
    """
    p = randint(len(a))
    a[p] = random_sample(2) * r


class Gene:
    def __init__(self, k, n, party, _n, r):
        """
        :param k: k개의 센트로이드
        :param n: n개의 유전자
        :param party: 정당 개수
        :param _n: 점의 개수
        :param r: 반경
        """
        self.make = Make(party, _n, 10000, r)

        self.genes = [random_sample((k, 2)) * r for x in range(n)]
        self.k = k
        self.sel = []
        self.r = r
        self.n = n

    def selection(self):
        """
        적합도를 바탕으로 선택 연산
        상위 10% + 랜덤 10%
        """
        arg = list(range(len(self.genes)))
        arg.sort(key=lambda x: self.make.cal(self.genes[x]))
        sel1 = arg[:self.n//10]
        sel2 = choice(arg, size=self.n//10)
        sel1.extend(sel2)
        self.sel = sel1
        shuffle(self.sel)

    def cross_child(self, a, b):
        """
        a, b에 대해 산술 교차를 계산
        10% 확률로 변이를 일으킨다.
        """
        a = np.asarray([[arith(e[0], b[w][0]), arith(e[1], b[w][1])] for w, e in enumerate(a)])
        # print(a)
        if random() * 100 < 10:
            mutation(a, self.r)
        return a

    def cross(self, n):
        """
        :param n: 짝을 지은 유전자는 n개의 후대 유전자 생성
        교차 연산으로 후대 유전자 생성
        """
        child = [
            self.cross_child(self.genes[self.sel[i]], self.genes[self.sel[len(self.sel) - i - 1]])
            for i in range(int(len(self.sel) / 2))
            for _ in range(n)
        ]
        return child

    def run(self, epoch):
        res = []
        gene = []

        for x in range(epoch):
            self.selection(0)
            child = self.cross(5)
            sor = sorted(self.genes, key=lambda x: self.make.cal(x))
            res.append(self.make.cal(sor[0]))
            self.genes = child
            self.genes.extend(sor[:10])
            self.genes.extend([sor[x] for x in choice(np.arange(len(sor)), size=10)])
            gene.append(sor[0])
            print(f'epoch {x+1}: {res[-1]}')
        ma = min(self.genes, key=lambda x: self.make.cal(x))
        res.append(self.make.cal(ma))
        gene.append(ma)
        return np.array(res), np.array(gene)



"""
:param k: k개의 centroid
:param n: n개의 유전자
:param party: 정당 개수
:param _n: 점의 개수
:param r: 반경
"""
def main(e):
    import matplotlib.pyplot as plt
    gene = Gene(15, 200, 6, 200, 100)
    print('------------ start ------------')
    res, genes = gene.run(e)
    plt.plot(list(range(len(res))), res)
    plt.plot(list(range(len(res))), res[:, 1] * 1000)
    plt.show()
    np.save('res1', res)
    np.save('genes', genes)
    np.save('pos', gene.make.pos)

if __name__ == '__main__':
    main(200)
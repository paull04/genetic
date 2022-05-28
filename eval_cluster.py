import numpy as np


def cal_cluster(centers, points, k, candi):
    arr = np.zeros([len(centers), k])
    for p in points:
        dis = [sum(x) for x in (centers - p) ** 2]
        where = np.argmin(dis)
        arr[where] += candi[where]
    """
    각 centroid마다 정당이 받은 투표수 계산
    """

    return arr


def eval_cluster(counts, votes):
    """
    의석수 계산과 득표율과의 차이 계산
    :param total: 전체 투표자
    :param votes: 전체 득표율
    :param counts: 지역별 득표수
    :return: 의석수와 득표율의 차이

    득표율과 의석 점유율 차이 계산
    """
    seats = np.zeros_like(votes)
    cnt = len(seats)
    for x in counts:
        idx = np.argmax(x)
        seats[idx] += 1
    #print(seats)
    #print(votes*cnt)
    dif = np.sum(np.abs(seats - votes * cnt))
    return dif


def is_alive(centers, points, num):
    arr = np.zeros(len(centers))
    for w, p in enumerate(points):
        dis = [sum(x) for x in (centers - p) ** 2]
        where = np.argmin(dis)
        arr[where] += sum(num[w])
    """
    최대 인구와 최소 인구의 차 계산
    
    
    첫번쨰 실험에 경우
    return ma - mi
    """
    mi = np.min(arr)
    ma = np.max(arr)
    #print(mi, ma)
    if ma < mi * 3:
        return 0
    return ma - mi * 3



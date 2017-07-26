# implement fcbf algorithm
import fs.infoTheoryUtil as itUtil
import fs.fileUtil as fUtil
import numpy as np


def alg_fcbf(dataset):
    m, n = dataset.shape    # m: number of samples, n: number of features + 1
    t_cor = []
    for i in range(n-1):
        t_cor.append(itUtil.su(dataset[:, i], dataset[:, n-1]))
    temp = [(value, index) for index, value in enumerate(t_cor)]
    temp.sort(reverse=True)
    print(temp)
    temp = [fc for fc in temp if fc[0] > temp[int((n-1) / np.log(n-1))][0]]
    list_best = []
    while temp:
        print("add --", temp[0][1])
        list_best.append(temp[0][1])
        for fc1 in temp[1:]:
            if itUtil.su(dataset[:, temp[0][1]], dataset[:, fc1[1]]) >= fc1[0]:
                temp.remove(fc1)
                print("remove --", fc1[1])
        temp = temp[1:]
    return list_best

if __name__ == "__main__":
    data = fUtil.read_data(r"../datasets/lung-cancer.data", 57)
    print(alg_fcbf(data, 0.))



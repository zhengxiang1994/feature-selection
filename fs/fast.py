# implement fast algorithm
# 1. construct a complete graph
# 2. construct a mst from the complete graph
# 3. partition the mst
# 4. choose the representative feature from each tree
import fs.fileUtil as fUtil
import fs.infoTheoryUtil as itUtil
import numpy as np


def alg_fast(dataset):
    m, n = dataset.shape  # m: number of samples, n: number of features + 1
    t_cor = []
    for i in range(n - 1):
        t_cor.append(itUtil.su(dataset[:, i], dataset[:, n - 1]))
    temp = [(value, index) for index, value in enumerate(t_cor)]
    temp.sort()
    threshold = temp[int(np.sqrt(n-1) * np.log(n-1))][0]
    temp = [fc for fc in temp if fc[0] > threshold]
    print(temp)
    # store the original sequence
    sequence = {after: before[1] for after, before in enumerate(temp)}
    print(sequence)
    adjacent_table = []
    for f1 in temp:
        for f2 in temp:
            if f1[1] == f2[1]:
                adjacent_table.append(float("inf"))
            else:
                adjacent_table.append(itUtil.su(dataset[:, f1[1]], dataset[:, f2[1]]))
    adjacent_table = np.array(adjacent_table).reshape(len(temp), len(temp))
    print(adjacent_table)
    # construct the mst
    mst_vertices = [0]
    mst_edges = []
    while len(mst_vertices) < len(temp):
        temp_edge = (0, 0)
        temp_vertex = 0
        for vertex1 in mst_vertices:
            for vertex2 in range(len(temp)):
                if (vertex1, vertex2) not in mst_edges and (vertex2, vertex1) not in mst_edges and vertex2 not in \
                        mst_vertices and adjacent_table[vertex1, vertex2] < adjacent_table[temp_edge[0], temp_edge[1]]:
                        temp_edge = (vertex1, vertex2)
                        temp_vertex = vertex2
        mst_vertices.append(temp_vertex)
        mst_edges.append(temp_edge)
        print("add vertex:", temp_vertex)
        print("add edge:", temp_edge)
        print("-" * 10)
    print(mst_vertices, mst_edges, sep="\n")
    # partition the mst
    rest_edges = []
    for edge in mst_edges:
        if adjacent_table[edge[0]][edge[1]] >= temp[edge[0]][0] or adjacent_table[edge[0]][edge[1]] >= temp[edge[1]][0]:
            rest_edges.append(edge)
    print(rest_edges)
    # choose the representative feature
    best_list = []
    while mst_vertices:
        redundant_list = [mst_vertices[0]]
        for edge in rest_edges:
            if edge[0] in redundant_list:
                redundant_list.append(edge[1])
            elif edge[1] in redundant_list:
                redundant_list.append(edge[0])
        print("--", redundant_list, "--")
        if len(redundant_list) == 1:
            best_list.append(sequence[redundant_list[0]])
        else:
            temp_best = redundant_list[0]
            for i in range(1, len(redundant_list)):
                if temp[redundant_list[i]][0] > temp[temp_best][0]:
                    temp_best = redundant_list[i]
            best_list.append(sequence[temp_best])
        mst_vertices = [vertex for vertex in mst_vertices if vertex not in redundant_list]
    return best_list

if __name__ == "__main__":
    data = fUtil.read_data(r"../datasets/lung-cancer.data", 57)
    # data = fUtil.read_data(r"../datasets/breast-cancer-wisconsin.data.txt", 11)
    print(alg_fast(data))




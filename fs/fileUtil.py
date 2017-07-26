# util of file dispose
import numpy as np


# read data from text
def read_data(filename):
    lines = []
    arff_file = open(filename, "r")
    for line in arff_file:
        if not (line.startswith("@")):
            if not (line.startswith("%")) and line.strip():
                arr = line.strip("\n").split(",")
                if "?" not in arr:
                    lines.append(arr)
    lines = np.array(lines)
    m, n = lines.shape
    for i in range(n):
        column = list(lines[:, i])
        if not column[0].isdigit():
            elements = list(set(column))
            elements.sort()
            new_column = [elements.index(s) for s in column]
            lines[:, i] = np.array(new_column)
        else:
            new_column = [int(s) for s in column]
            lines[:, i] = np.array(new_column)
    return lines


if __name__ == "__main__":
    print(read_data("../datasets/lung-cancer.data"))
    print(read_data("../datasets/breast-cancer-wisconsin.data.txt"))



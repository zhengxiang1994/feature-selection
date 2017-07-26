# util of information theory
import numpy as np


# calculate the entropy H(f)
def entropy(f):
    ff = set(f)
    ety = 0.
    for value in ff:
        count = 0
        for i in range(f.size):
            if value == f[i]:
                count += 1
        ety += float(count/f.size) * (np.log2(float(count/f.size)))
    return -ety


# calculate H(f1|f2)
def conditional_entropy(f1, f2):
    ff1 = set(f1)
    ff2 = set(f2)
    ce = 0.
    for value2 in ff2:
        count2 = 0
        for i in range(f2.size):
            if value2 == f2[i]:
                count2 += 1
        p_f2 = float(count2 / f2.size)
        temp1 = 0.
        for value1 in ff1:
            count1 = 0
            for j in range(f1.size):
                if value1 == f1[j] and value2 == f2[j]:
                    count1 += 1
            if count1 != 0:
                # calculate condition probability p(f1|f2)
                p_f1f2 = float(count1 / f1.size) / p_f2
                temp1 += (p_f1f2 * np.log2(p_f1f2))
                # print temp1
        ce += p_f2 * temp1
    return -ce


# calculate information gain IG(f1|f2)
def ig(f1, f2):
    return entropy(f1) - conditional_entropy(f1, f2)


# calculate symmetrical uncertainty SU(f1, f2)
def su(f1, f2):
    return 2 * ig(f1, f2) / (entropy(f1) + entropy(f2))



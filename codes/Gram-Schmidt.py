# Gram-Schmidt algorithm produces an prthonormal collection of vectors
# GS正交化算法, 注意，正交的是行向量，若要正交列向量，需要转置一下
# GS算法不是为了求一组矩阵的基，而是用来正则化一组基向量，顺便测试他是否满足一个基


import numpy as np


def GramSchmidt(array):
    A = array.astype(np.float64)
    Q = np.zeros_like(A, dtype=np.float64)
    for k, a in enumerate(A):
        q = a
        for i in range(k):
            qi = Q[i]
            q -= np.dot(qi, a)*qi
        if q.any() == 0:
            raise ValueError('输入的向量组不满足线性无关')
        else:
            q = q/np.sum(q**2)**0.5
            Q[k] = q
    return Q

# fuck! Debug了一个多小时，发现原来是数据类型的问题！！！！

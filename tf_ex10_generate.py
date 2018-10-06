# coding: utf-8
# tensorflow_ex10的模块化实现————生成数据集合
# 0. 导入模块， 生成模拟数据集合
import numpy as np
import matplotlib.pyplot as plt
seed = 2
def generateds():
    # 基于seed产生随机数
    rdm = np.random.RandomState(seed)
    # 随机数返回300x2的矩阵，表示300组坐标点作为输入数据集
    X = rdm.randn(300, 2)
    # 从X这300X2的矩阵中取出一行，判断如果两个坐标平方和小于2，给Y赋值为1，否则赋值为0
    # 作为输入数据集的标签（正确答案）
    Y_ = [int(x0**2+x1**2<2) for (x0, x1) in X]
    # 遍历Y中的每个元素，1赋值'red'，0赋值'blue'，这样可视化显示时可以直观区分
    Y_c = [['red' if y else 'blue'] for y in Y_]
    # 对数据集X和标签Y进行shape整理，第一个元素为-1表示，随第二个参数计算得到，第二个元素表示多少列，
    # 把X整列成n行2列，把Y整理为n行1列
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)
    # 用plt.scatter画出数据X各行中第0列和第1列元素的点，用各行Y_c对应的值表示颜色
    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
    plt.show()
    return X, Y_, Y_c

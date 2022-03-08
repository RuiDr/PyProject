# -*- coding: utf-8 -*-
# @Author  : dr
# @Time    : 2022/3/8 14:57
# @Function:K-近邻算法
from numpy import *
import operator


# 创建数据
def createDataSet():
    # 4组数据，每组两个属性和特征值
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'B', 'B', 'B']
    return group, labels


# inX: 用于分类的输入向量; dataSet: 训练样本集; labels:标签向量
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 距离求解 tile 复制给定内容，并生成指定行列的矩阵，得到x1-x2，y1-y2组成的矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 矩阵中每个元素平方即得到(x1-x2)的平方
    sqDiffMat = diffMat**2
    # 按行求和，即求出(x1-x2)的平方 + (y1-y2)的平方
    sqDistances = sqDiffMat.sum(axis=1)
    # 开根号，求得距离
    distances = sqDistances**0.5
    # 按照距离排序，将distances中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    sortedDisIndicies = distances.argsort()
    classCount = {}
    # 求频率
    for i in range(k):
        votelabel = labels[sortedDisIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    # sorted 按照升序返回一个列表
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    print(classify0([0, 0.1], group, labels, 3))

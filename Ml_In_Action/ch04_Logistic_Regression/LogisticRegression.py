# -*- coding: utf-8 -*-
# @Author  : dr
# @Time    : 2022/3/16 10:22
# @Function: Logistic回归梯度上升优化算法
from math import exp
from numpy import*
import numpy as np
import matplotlib.pyplot as plt


# 解析文件里面的数据
def loaddataset():
    dataMat = []
    ldLabelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        ldLabelMat.append(int(lineArr[2]))
    return dataMat, ldLabelMat


def sigmiod(inx):
    return 1.0/(1+exp(-inx))


# datamathin是一个2维的numpy数组，列属性，行样本
def gradascent(datamatin, classlabels):
    datamatrix = mat(datamatin)
    # 行转列
    labelmat = mat(classlabels).transpose()
    m, n = shape(datamatrix)
    # 移动步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    wgt = ones((n, 1))
    for k in range(maxCycles):
        # 真实值与预测类别的差值，所有样本
        h = sigmiod(datamatrix * wgt)
        error = labelmat - h
        wgt = wgt + alpha * datamatrix.transpose() * error
    return wgt


# 随机梯度上升算法
def stocgradascent0(datamatrix, classlabels):
    m, n = shape(datamatrix)
    alpha = 0.01
    sgsWeights = ones(n)
    for i in range(m):
        # 第i个样本点的fx值，float内为第一个样本点的w*x
        h = sigmiod(sum(datamatrix[i] * sgsWeights))
        # 第i个样本的真实值与预测值误差，用于回归系数调整
        error = classlabels[i] - h
        # 回归系数调整，只用第i个样本点更新回归系数w
        sgsWeights = sgsWeights + alpha * error * datamatrix[i]
    return sgsWeights


# 改进随机梯度上升算法
def stocgradascent1(datamatrix, classlabels, numiter=150):  # 改进的随机梯度上升算法
    # m个样本，n维特征
    m, n = shape(datamatrix)
    # 初始化回归系数
    sgt1Weights = ones(n)
    for j in range(numiter):
        # 样本点索引，用于存储尚未用于更新系数的样本点的索引
        dataIndex = list(range(m))
        # 遍历所有样本点
        for i in range(m):
            # 第j次迭代的第i小迭代
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机抽取样本点进行系数更新
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmiod(sum(datamatrix[randIndex] * sgt1Weights))
            error = classlabels[randIndex] - h
            sgt1Weights = sgt1Weights + alpha * error * datamatrix[randIndex]
            del (dataIndex[randIndex])
    return sgt1Weights


# 画出数据集和Logistic回归最佳拟合直线的函数
def plotbestfit(pbtweights):
    pbtDataMat, pbtLabelMat = loaddataset()
    pbtArr = array(pbtDataMat)
    pbtN = shape(dataArr)[0]
    pbtXcord1 = []
    pbtYcord1 = []
    pbtXcord2 = []
    pbtYcord2 = []
    # 将不同分类的点放入不同的集合，用不同的颜色表示
    for i in range(pbtN):
        if int(pbtLabelMat[i]) == 1:
            pbtXcord1.append(pbtArr[i, 1])
            pbtYcord1.append(pbtArr[i, 2])
        else:
            pbtXcord2.append(pbtArr[i, 1])
            pbtYcord2.append(pbtArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pbtXcord1, pbtYcord1, s=30, c='blue', marker='s')
    ax.scatter(pbtXcord2, pbtYcord2, s=30, c='red')
    # x坐标-3到3，间隔0.1
    x = arange(-3.0, 3.0, 0.1)
    y = (-pbtweights[0]-pbtweights[1] * x)/pbtweights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataArr, labelMat = loaddataset()

    weights1 = gradascent(dataArr, labelMat)
    print(weights1)
    plotbestfit(weights1.getA())

    dataArr, labelMat = loaddataset()
    weights2 = stocgradascent0(array(dataArr), labelMat)
    print(weights2)
    plotbestfit(weights2)

    dataArr, labelMat = loaddataset()
    weight3 = stocgradascent1(array(dataArr), labelMat, 500)
    print(weight3)
    plotbestfit(weight3)



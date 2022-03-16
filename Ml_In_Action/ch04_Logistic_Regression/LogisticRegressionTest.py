# -*- coding: utf-8 -*-
# @Author  : dr
# @Time    : 2022/3/16 16:24
# @Function:从疝气病症预测病马的死亡率
# 步骤：1 收集数据，2 准备数据(填充缺失值)，3分析数据(可视化并观察数据),
# 4训练算法(寻找最佳系数)，5测试算法：为了量化回归结果，需要观察错误率，
# 根据错误率决定是否回退到训练阶段，通过改变迭代次数和步长等参数来得到更好的回归次数
# 6使用算法
from math import exp
from numpy import*


def sigmiod(inx):
    return 1.0/(1+exp(-inx))


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


# 回归系数和特征向量作为输入来计算对应的sigmoid值，大于0.5返回1，否则返回0
def classifyvector(inx, weights):
    prob = sigmiod(sum(inx * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# 数据处理
def colictest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append((float(currLine[i])))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocgradascent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyvector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is : %f" % errorRate)
    return errorRate


# 调用colictest 10次并且结果的平均值
def multitest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colictest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


if __name__ == '__main__':
    multitest()




















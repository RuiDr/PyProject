# -*- coding: utf-8 -*-
# @Author  : dr
# @Time    : 2022/3/8 14:57
# @Function:K-近邻算法
from numpy import *
import operator
import os
import matplotlib
import matplotlib.pyplot as plt


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


# 将数据处理成分类器所需数据
def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        # 移除字符串头尾指定的字符(默认空格或换行符)
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 使用Matplotlib创建散点图
def createfig(datingdatamat, datinglabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingdatamat[:, 1], datingdatamat[:, 2], 15.0 * array(datinglabels), 15.0 * array(datinglabels))
    plt.show()


# 数值归一化，将特征值转化为0到1的区间
def autonorm(dataset):
    # 从列中选取最小值
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataset))
    m = dataset.shape[0]
    normDataSet = dataset - tile(minVals, (m, 1))
    # 特征值相除，numpy中矩阵除法使用linalg.solve(matA, ,matB)
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 分类器针对约会网站的测试代码
def datingcalsstest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('E:\projects\python\PyProject\Ml_In_Action\ch01_KNN\datingTestSet2.txt')
    normMat, ranges, minVals = autonorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classfierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is:%d", classfierResult, datingLabels[i])
        if classfierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f", (errorCount/float(numTestVecs)))


# 将图像格式化为一个向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


# 使用k-近邻算法识别手写数字系统
def handwritingclasstest():
    hwLabels = []
    trainingFileList = os.listdir('E:\\projects\\python\\PyProject\\Ml_In_Action\\ch01_KNN\\trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        # 从文件名解析分类数字
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('E:\\projects\\python\PyProject\\Ml_In_Action\\ch01_KNN\\trainingDigits/%s' % fileNameStr)
    # 测试集
    testFileList = os.listdir('E:\\projects\\python\\PyProject\\Ml_In_Action\\ch01_KNN\\testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('E:\\projects\\python\\PyProject\\Ml_In_Action\\ch01_KNN\\testDigits/%s' % fileNameStr)
        classifilerResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is:%d", classifilerResult, classNumStr)
        if classifilerResult != classNumStr:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is:%f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    # 分类器针对约会网站的测试。最后输出错误率
    # datingcalsstest()
    # 手写系统
    handwritingclasstest()
    # group, labels = createDataSet()
    # print(classify0([0, 0.1], group, labels, 3))
    # returnMat, classLabelVector = file2matrix('E:\projects\python\PyProject\Ml_In_Action\ch01_KNN\datingTestSet2.txt')
    # createfig(returnMat, classLabelVector)

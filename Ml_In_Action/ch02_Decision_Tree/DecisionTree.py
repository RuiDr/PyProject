# -*- coding: utf-8 -*-
# @Author  : dengrui
# @Time    : 2022/3/10 11:20
# @Function: 决策树
import operator
from math import log


# 计算可定数据集的香农熵
def calcshannonent(shannondataset):
    # 计算实例总数
    numEntries = len(shannondataset)
    labelCounts = {}

    # 统计键值出现的次数
    for featVec in shannondataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # 计算类别出现概率
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照给定特征划分数据集,当我们按照摸一个特征划分数据集时，就需要将所有符合要求的元素提取出来
# 传入三个参数，第一个是我们的数据集，第二个是要依据某个特征来划分数据集
def splitdataset(datasets, axis, value):
    retDataSets = []
    for featVec in dataset:
        # 判断特征与我们指定的特征值是否相等
        if featVec[axis] == value:
            # 除去这个特征创建子特征
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSets.append(reducedFeatVec)
    return retDataSets


# 选择最好的数据集划分方式
def choosebestfeaturetosplit(choosedataset):
    # 计算特征值的数量
    leng = choosedataset[0]
    numFeatures = len(leng) - 1
    # 计算整个数据集的原始香农熵
    baseEntropy = calcshannonent(choosedataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in choosedataset]
        # 得到列表中唯一属性值
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            # 对每个特征划分一次数据集
            subDataSet = splitdataset(choosedataset, i, value)
            # 求唯一特征值的熵和
            prob = len(subDataSet)/float(len(choosedataset))
            newEntropy += prob * calcshannonent(subDataSet)
        infoGain = baseEntropy - newEntropy
        # 计算最好的信息增益
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 创建数据
def createdataset():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    rlabels = ['no surfacing', 'flippers']
    return dataSet, rlabels


# 统计分量名称数量
def majoritycnt(classlist):
    classCount = {}
    for vote in classlist:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter,reverse=True)
    return sortedClassCount[0][0]


# 创建决策树
def createtree(ctdataset, ctlabels):
    # 数据集中所有类标签
    classList = [example[-1] for example in ctdataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(ctdataset[0]) == 1:
        return majoritycnt(classList)
    bestFeat = choosebestfeaturetosplit(ctdataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in ctdataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createtree(splitdataset(ctdataset, bestFeat, value), subLabels)
    return myTree


if __name__ == '__main__':
    dataset, labels = createdataset()
    bestfeature = choosebestfeaturetosplit(dataset)
    # retDataSet = splitdataset(dataset, 1, 1)
    print(bestfeature)

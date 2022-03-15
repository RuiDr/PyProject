# -*- coding: utf-8 -*-
# @Author  : dengrui
# @Time    : 2022/3/15 10:17
# @Function: 朴素贝叶斯算法实现
from numpy import*


# 创建样本
def loaddataset():
    """
    postingList: 进行词条切分后的文档集合
    classVec:类别标签
    使用伯努利模型的贝叶斯分类器只考虑单词出现与否（0，1）
    """
    # postingList: 进行词条切分后的文档集合
    postingList = [['my', 'dog',   'help'],
                 ['not',  'dog', 'stupid'],
                 ['my',  'so', 'love']]
    # 类别标签，1代表侮辱性文字，0代表正常言论
    classVec = [0, 1, 0]
    return postingList, classVec


# 创建一个包含在所有文档中出现的不重复词的列表
def createvocablist(cvldataset):
    vocabSet = set([])
    for document in cvldataset:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 词集模型
# 输入参数为词汇表及某个文档，输出的是文档向量，
# 向量的每一元素为1或0，分别表示词汇表中的单词在输入文档中是否出现
def setofwords2vec(vocablist, inputset):
    # 创建一个与词汇表等长的向量，并将其元素都设置为0
    sow2ReturnVec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            sow2ReturnVec[vocablist.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return sow2ReturnVec


# 词袋模型
def bagofwords2vec(vocablist, inputset):
    # 创建一个与词汇表等长的向量，并将其元素都设置为0
    returnVec = [0] * len(vocablist)
    for word in inputset:
        if word in vocablist:
            returnVec[vocablist.index(word)] += 1
    return returnVec


# 输入文档矩阵trainmatrix以及每篇文档类别标签所构成的向量traincategory
def trainnbo(trainmatrix, traincategory):
    numTrainDocs = len(trainmatrix)
    numWords = len(trainmatrix[0])
    # 在输入文本中，侮辱性(1)占总类别的概率，即p(y1)的概率，由于为二分类，所以正常言论(0)为1-pAbusive
    pAbusive = sum(traincategory)/float(numTrainDocs)
    # 初始化概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 如果其中一个概率值为0，那么最后的乘积也为0，所以将所有词的出现数初始化为1，并将分母初始化为2
    p0Denom = 2.0
    p1Denom = 2.0
    # 遍历训练集trainMatrix中所有的文档，一旦某个词(侮辱性或正常词语)在某一文档中出现，
    # 则对应（P1Num或P0Num）的个数+1,而且在所有的文档中，该文档的总词数也相应加1
    # 求每个属性在各个类别中的数量，即p(x/yi)其中x在这里表示每个不重复的单词
    for i in range(numTrainDocs):
        if traincategory[i] == 1:
            p1Num += trainmatrix[i]
            p1Denom += sum(trainmatrix[i])
        else:
            p0Num += trainmatrix[i]
            p0Denom += sum(trainmatrix[i])
    # 每个元素除以该类别的总词数,由于大部分因子都非常小，所以程序会下溢或者得不到正确答案，可以通过求对数解决这个问题
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifynb(vec2classify, p0vec, p1vec, pclass1):
    # p(x/yi)*p(yi),这里求sum是防止溢出，求得对数值
    p1 = sum(vec2classify * p1vec) + log(pclass1)
    p0 = sum(vec2classify * p0vec) + log(1.0 - pclass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingnb():
    listOPosts, listClasses = loaddataset()
    myVocabList = createvocablist(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setofwords2vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainnbo(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setofwords2vec(myVocabList, testEntry))
    print("testEntry: %s classified as: %s ", testEntry, classifynb(thisDoc, p0V, p1V, pAb))


if __name__ == '__main__':
    # # 创建样本
    # listOPosts, listClasses = loaddataset()
    # # 去除重复词汇
    # myVocabList = createvocablist(listOPosts)
    # # 使用词汇表或者想要检查的所有单词作为输入
    # returnVec = setofwords2vec(myVocabList, listOPosts[0])
    # trainMat = []
    # # 统计词汇表
    # for postinDoc in listOPosts:
    #     # 词汇表中的单词在出入文本中是否出现，出现则为1，不出现则为0
    #     trainMat.append(setofwords2vec(myVocabList, postinDoc))
    # # 概率计算
    # p0V, p1V, pAb = trainnbo(trainMat, listClasses)
    # print("p0V:", p0V)
    # print("p1V:", p1V)
    # print("pAb:", pAb)
    # print(returnVec)
    # 测试
    testingnb()


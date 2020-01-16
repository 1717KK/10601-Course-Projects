#!/usr/bin/env python3
import sys
import csv
import math
import copy


##### build decision tree
        
    
# check if the dataset is perfectly classified
def ifUnambiguous(d):

    for key in d:
        if d[key] == 0:
            return True
    return False

# return the value of leaf
def perfectClassifier(d):
    for key in d:
        if d[key] != 0:
            return key

def majorityVote(d):
    maxNum = -math.inf
    maxValue = None
    for key in d:
        if d[key] > maxNum:
            maxNum = d[key]
            maxValue = key
    return maxValue

def printTree(data, depth, attribute, value, y):
    
    d = {y[0]: 0, y[1]: 0}
    for i in range(len(data)):
        if data[i][-1] not in d:
            d[data[i][-1]] = 1
        else:
            d[data[i][-1]] += 1
    
    lst = []
    for key in d:
        each = [key, d[key]]
        lst += each
    
    if depth == 0:
        print( "[" + str(lst[1]) + " "+ lst[0] + " /" + str(lst[3]) + " " + lst[2] + "]")
        
    else:
        print("| " * depth + attribute + " = " + value + ": [" + str(lst[1]) + " "+ lst[0] + " /" + str(lst[3]) + " " + lst[2] + "]")
    
    return d

# data: value (2D List)
# remainingAttr: name of columns (List)
def decisionTreeTrain(data, y, remainingAttr, depth, maxDepth, attribute, value):
    #Node(self, attribute, valueLeft, valueRight, label, left, right, depth)
    root = Node(attribute, None, None, None, None, None, depth)
    
    # print the tree
    # d: a dictionay to store the remaining dataset
    d = printTree(data, depth, attribute, value, y)
    
    #base case: data is unambiguous, no need to split further
    if ifUnambiguous(d):  
        root.label = perfectClassifier(d)
        return root
        
    #base case: no more remaining features, cannot split further
    #or stop grow the tree beyond the max-depth
    elif len(remainingAttr) == 1 or depth >= maxDepth: 
        root.label = majorityVote(d)
        return root
    
    else:
        leftSubset = [] 
        rightSubset = [] 
        total = [remainingAttr] + data
        
        (attribute, idx, highestMI) = splitAttribute(total) # name of the attribute, index of the attribute, highest MI
        
        if highestMI == 0:
            root.label = perfectClassifier(d)
            return Node(attribute, valueLeft, valueRight, root.label, root.left, root.right, depth)
            
        
        elif highestMI > 0:
        
            (valueLeft, leftSubset, valueRight, rightSubset) = generateSubset(data, idx)
            
            recRemainAttr = copy.deepcopy(remainingAttr)
            recRemainAttr.remove(attribute)
            
            root.left = decisionTreeTrain(leftSubset, y, recRemainAttr, depth + 1, maxDepth, attribute, valueLeft)
            
            if valueRight != None:
                
                root.right = decisionTreeTrain(rightSubset, y, recRemainAttr, depth + 1, maxDepth, attribute, valueRight)
                
                #Node(self, attribute, valueLeft, valueRight, label, left, right, depth)
                return Node(attribute, valueLeft, valueRight, None, root.left, root.right, depth)
                
            else:
                return Node(attribute, valueLeft, valueRight, None, root.left, None, depth)
        

def generateSubset(data, highestIdx):
    
    d = dict()
    for i in range(len(data)): # the same as loop the remaining attribute            
        if data[i][highestIdx] not in d:
            d[data[i][highestIdx]] = []

    for i in range(len(data)):
        row = []
        for j in range(len(data[0])):
            if j != highestIdx:
                row += [data[i][j]]
        d[data[i][highestIdx]] += [row]

    subsetList = []
    for key in d:
        subsetList += [key]
        subsetList += [d[key]]

    if len(subsetList) == 4:
        valueLeft = subsetList[0]
        left = subsetList[1]
        valueRight = subsetList[2]
        right = subsetList[3]
        
        return (valueLeft, left, valueRight, right)
        
    elif len(subsetList) == 2:
        valueLeft = subsetList[0]
        left = subsetList[1]
        
        return (valueLeft, left, None, None)
    
def splitAttribute(train):
    attribute = train[0]
    train = train[1:]
    highestMI = -float(math.inf)
    highestIdx = 0
    colName = ""
    for j in range(len(train[0])-1): #col except the last one
        x = []
        y = []
        for i in range(len(train)):
            x += [train[i][j]]
            y += [train[i][-1]]
        MI = float(calMutualInfo(x, y))
        
        if MI > highestMI:
            highestMI = MI
            highestIdx = j
            colName = attribute[j]
    
    return (colName, highestIdx, highestMI)

    
class Node(object):
    def __init__(self, attribute, valueLeft, valueRight, label, left, right, depth):
        self.attribute = attribute
        self.label = label
        self.valueLeft = valueLeft
        self.valueRight = valueRight
        self.left = left
        self.right = right
        self.depth = depth
        

        
    
##### calculate entropy, conditional entropy, mutual information

def calMutualInfo(x, y):
    
    hy = calEntropy(y)
    hyx = calConEntropy(x, y)
    
    return hy - hyx
    

def calEntropyHelper(data):
    # a dictionary to store the number of each class
    d = dict()
    for ele in data:
        if ele not in d:
            d[ele] = 1
        else:
            d[ele] += 1
    return d


# calculate entropy H(Y)
def calEntropy(data):
    
    d = calEntropyHelper(data)
    total = 0
    entropy = 0
    for key in d:
        total += d[key]
    for key in d:
        partition = d[key] / total
        d[key] = partition
    for key in d:
        entropy -= d[key] * math.log(d[key], 2)  
    return entropy
    

# calculate conditional entropy H(Y|X=v)
def calConEntropyHelper(x, y):
    
    d = dict()
    for i in range(len(x)):
        if x[i] not in d:
            d[x[i]] = [y[i]]
        else:
            d[x[i]] += [y[i]]
    
    
    #store the partition of each value of X
    numXList = []
    totalX = 0
    for key in d:
        numX = len(d[key])
        totalX += numX
        numXList += [numX]
    
    for i in range(len(numXList)):
        numXList[i] = numXList[i]/totalX
    
    #calculate the conditional probability
    for key in d:
        dictX = dict()
        for ele in d[key]:
            if ele not in dictX:
                dictX[ele] = 1
            else:
                dictX[ele] += 1
        d[key] = dictX
    
    for key in d:
        total = 0
        for ele in d[key]:
            total += d[key][ele]
        for ele in d[key]:
            d[key][ele] = d[key][ele]/total
        
    return [d, numXList]
    
# calculate conditional entropy H(Y|X)
def calConEntropy(x, y):
    
    [d, numXList] = calConEntropyHelper(x, y)
    tmpConEntropyList = [] #conditional entropy H(Y|X=v)
    for key in d:
        tmpConEntropy = 0
        for partition in d[key]:
            tmpConEntropy -= d[key][partition] * math.log(d[key][partition], 2)
        tmpConEntropyList += [tmpConEntropy]
    
    # conditional entropy H(Y|X)
    conEntropy = 0
    for i in range(len(numXList)):
        probX = numXList[i]
        conEntropy += probX * tmpConEntropyList[i]
        
    return conEntropy

#### predict label

def predictLabelHelper(row, attribute, root):
    
    if root.label != None:
        return root.label
    else:
    
        for i in range(len(row)):
            if attribute[i] == root.attribute and row[i] == root.valueLeft:
                return predictLabelHelper(row, attribute, root.left)
            elif attribute[i] == root.attribute and row[i] == root.valueRight:
                return predictLabelHelper(row, attribute, root.right)
    
    
    
def predictLabel(data, root):
    
    attribute = data[0]
    labelList = []
    for row in data[1:]:
        predictedLabel = predictLabelHelper(row, attribute, root)
        labelList += [predictedLabel]
    return labelList
    
#### calculate the error rate

def calErrorRate(labelList, data):

    y = []
    for i in range(len(data)):
        y += [data[i][-1]]
        
    num = 0
    total = len(labelList)

    for i in range(total): 
        if labelList[i] != y[i]:
            num += 1
    errorRate = num/total
    
    return errorRate
    
#### prepare the data

# store the value of y
def preData(data):
    y = []
    for i in range(len(data)):
        if data[i][-1] not in y:
            y += [data[i][-1]]
    return y
    
#### read the files

trainFile = 
testFile = 

if __name__ == '__main__':
    
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
    maxDepth = int(sys.argv[3])
    labelTrain = sys.argv[4]
    labelTest = sys.argv[5]
    metrics = sys.argv[6]
    
    
    with open(str(trainFile), "rt") as f1:
        tsvF1 = csv.reader(f1)
        data1 = [row1 for row1 in tsvF1]
        trainData = []
        for row in data1:
            line = row[0]
            trainData += [line.split('\t')]
        f1.close()
    
    with open(str(testFile), "rt") as f2:
        tsvF2 = csv.reader(f2)
        data2 = [row2 for row2 in tsvF2]
        testData = []
        for row in data2:
            line = row[0]
            testData += [line.split('\t')]
        f2.close()

#### test the files

    #training dataset
    
    yTrain = preData(trainData[1:])
    #data, class, allAttributeName, depth, maxDepth, target 
    rootTrain = decisionTreeTrain(trainData[1:], yTrain, trainData[0], 0, maxDepth, None, None)
    labelTrainList = predictLabel(trainData, rootTrain)
    errorRateTrain = calErrorRate(labelTrainList, trainData[1:])

    
    yTest = preData(testData[1:])
    rootTest = decisionTreeTrain(testData[1:], yTest, testData[0], 0, maxDepth, None, None)
    labelTestList = predictLabel(testData, rootTest)
    errorRateTest = calErrorRate(labelTestList, testData[1:])


#### write the results into files
    
    with open(str(labelTrain), "wt") as f:
        labelTrain = ""
        for l in labelTrainList:
            labelTrain += l
            labelTrain += "\n"
        f.write(labelTrain)
        
    with open(str(labelTest), "wt") as f:
        labelTest = ""
        for l in labelTestList:
            labelTest += l
            labelTest += "\n"
        f.write(labelTest)
        
    with open(str(metrics), "wt") as f:
        result = "error(train): " + str(errorRateTrain) + "\n" + "error(test): " + str(errorRateTest)
        f.write(result)
        
    print("written!")
        
        
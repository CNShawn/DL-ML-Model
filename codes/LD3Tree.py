import pandas as pd
import numpy as np
import math
# ld3 algorithm tree implemented by trie
# ld3 决策树算法，由字典树实现

class LD3Tree:
    def __init__(self, data, features):
        self.root = {}
        self.data = data
        self.features = features

    def creatTree(self):
        labelList = [sp[-1] for sp in self.data]
        if len(set(labelList))==1:
            return labelList[-1]
        if len(self.data[0])==2:
            return max(labelList, key=labelList.count)
        bestFeatIndex = self.maxEntFeature()
        bestFeat = self.features[bestFeatIndex]
        del(self.features[bestFeatIndex])
        self.root[bestFeat] = {}
        bestFeatVal = set([sp[bestFeatIndex] for sp in self.data])
        for val in bestFeatVal:
            subfeatures = self.features[:]
            self.root[bestFeat][val] = LD3Tree(self.SpiltbyAttribute(self.data, val, bestFeatIndex), subfeatures).creatTree()
        return self.root

    def ShannonEnt(self, data):
        # g(D, A) = H(D) - H(D|A)
        # if you need to calculate the H(D), all you need is passing the dataset
        # else if you want to calculate the H(D|A), you should pass the data spilted by attributes in different features
        # return is positive, remember to add a '-'
        labelCounts = {}
        for sample in data:
            currentLabel = sample[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1

        shannonent = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/len(data)
            shannonent += prob*math.log(prob, 2)
        return shannonent

    def SpiltbyAttribute(self, data, val, index):
        subdata = []
        for sample in data:
            if sample[index] == val:
                cursample = sample[:index]
                cursample.extend(sample[index+1:])
                subdata.append(cursample)
        return subdata

    def maxEntFeature(self):
        baseEnt = -self.ShannonEnt(self.data)
        bestgain = 0.0
        bestfeature = -1
        for i in range(len(self.features)):
            values = [val[i] for val in self.data]
            valueset = set(values)
            newEnt = 0.0
            for val in valueset:
                subdata = self.SpiltbyAttribute(self.data, val, i)
                prob = len(subdata)/float(len(self.data))
                shannonent = self.ShannonEnt(subdata)
                newEnt -= prob*shannonent
            gain = baseEnt-newEnt

            if gain > bestgain:
                bestgain = gain
                bestfeature = i
            return bestfeature


# tree = LD3Tree(data, features).creatTree()
# print(tree)

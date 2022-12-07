import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
import pickle
import random
from tqdm import tqdm


def transfrom_to_train(data: pd.DataFrame):
    data_ = data.to_numpy().tolist()
    attributeList = data.columns.values.tolist()

    return data_, attributeList
class decision_tree(object):


    def GetClassInfo(self,Dataset):
        ClassInfo = {}
        for item in Dataset:
            # check答案项目
            if item[-1] not in ClassInfo.keys():
                ClassInfo[item[-1]] = 1
            else:
                ClassInfo[item[-1]] += 1
            classInfo = dict(sorted(ClassInfo.items(), key=operator.itemgetter(1), reverse=True))
        return classInfo

    def maxclass(self,classinfo):
        max = list(classinfo.keys())[0]
        return max

    def compute_entropy(self,Dataset):
        classinfo  = self.GetClassInfo(Dataset)
        Ent = 0
        amount = 0
        p = []

        for _,value in classinfo.items():
            p.append(value)
            amount += value
        for p_k in p :
            Ent -= p_k/amount*np.log(p_k/amount)

        return Ent

    def getgainNPartition(self,Dataset,featureindex,featurelist):
        gain = self.compute_entropy(Dataset)
        lenth_dataset = len(Dataset)

        NPartition = {}
        for dataitem in Dataset:

            if dataitem[featureindex] not in NPartition.keys():
                NPartition[dataitem[featureindex]] = [dataitem]
            else:
                NPartition[dataitem[featureindex]].append(dataitem)

        lenth = []
        Ent = []
        amount = 0

        for _,subdataset in NPartition.items():
            Ent.append(self.compute_entropy(subdataset))
            lenth.append(len(subdataset))
            amount += len(subdataset)
        for i in range(len(Ent)):
            gain -= lenth[i]/lenth_dataset * Ent[i]

        return gain,NPartition

    def CreateDecisionTree(self,Dataset,featurelist):
        lenth_dataset = len(Dataset)
        classinfo = self.GetClassInfo(Dataset)
        Tree = {}
        #给定的属性集为空 ---- 不能划分
        if len(featurelist) == 0:
            # print(self.maxclass(classinfo))
            return self.maxclass(classinfo)

        #给定的数据集所有label都相同 ---- 无需划分
        for key,value in classinfo.items():
            if value == lenth_dataset:
                return key
                break
        #样本在属性集上取值都相等 ---- 无法划分
        temp = Dataset[0][:-1]
        sment = 0
        for dataitem in Dataset[:-1]:
            if dataitem == temp:
                sment += 1
        if sment == lenth_dataset:

            return self.maxclass(classinfo)
        # 选择最佳划分属性
        bestIdex = 0
        bestGain = 0
        bestNPartition = {}

        for indedx in range(len(featurelist)):
            gain,NPartition = self.getgainNPartition(Dataset,indedx,featurelist)
            if gain > bestGain:
                bestIdex = indedx
                bestGain = gain
                bestNPartition = NPartition

        attr_Name = featurelist[bestIdex]
        # print(type(featurelist))
        del featurelist[bestIdex]

        for _,vallist in bestNPartition.items():
            for i in range(len(vallist)):
                temp = vallist[i][:bestIdex]
                temp.extend(vallist[i][bestIdex+1:])
                vallist[i] = temp
        # # 为了方便后面建子树，将此时的attr对应的那列去除
        # 根据属性的值，建立分叉节点
        # if bestNPartition.values() == []:
        #     break

        Tree[attr_Name] = {}
        if not bestNPartition:
            # print(bestNPartition.values())
            return self.maxclass(classinfo)
        # print(featurelist)
        for keyAttrVal, valDataset in bestNPartition.items():

            # 因为python对iterable list对象的传参是按地址传参，会改变attributeList的值
            # 所以在传attributeList参数的时候，创建一个副本，就相当于按值传递了
            subLabels = featurelist[:]

            # valDataset是已去除attr的data，attributeList是已去除attr的attributeList
            Tree[attr_Name][keyAttrVal] = self.CreateDecisionTree(valDataset, subLabels)
        return Tree

    def vote(self,predic_label):
        vote_dict = {}
        for result in predic_label:

            if result not in vote_dict.keys():
                vote_dict[result] = 1
            else:
                vote_dict[result] += 1

            sort_vote_disc = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
            return sort_vote_disc[0][0]



    def Predict(self,DataSet, testArrtList, decisionTree):
        predicted_label = []

        for dataItem in DataSet:
            cur_decisionTree = decisionTree
            # 如果root就是叶子结点leaf
            if type(cur_decisionTree) == set:  # 例如：{'N'}
                node = list(cur_decisionTree)
            else:
                if  type(cur_decisionTree) == np.float:
                    node = cur_decisionTree
                else:
                    node = list(cur_decisionTree.keys())[0]
                # 只要temp处在attributeList，说明当前处在树枝结点(非叶子)上, 否则处在叶子结点
                while node in testArrtList:

                    try:

                        cur_index = testArrtList.index(node)  # 0 2
                        cur_element = dataItem[cur_index]  # 0 0
                        cur_decisionTree = cur_decisionTree[node][cur_element]  # {'student': {0: 'N', 1: 'Y'}} N
                        # print(cur_decisionTree)
                    except:

                        root = list(cur_decisionTree[node].keys())
                        cur_element = self.vote(root)
                        cur_decisionTree = cur_decisionTree[node][cur_element]


                    if type(cur_decisionTree) == dict:
                        node = list(cur_decisionTree.keys())[0]  # student
                    else:
                        node = cur_decisionTree


            # print(node)
            predicted_label.append(node)
        return predicted_label





class random_forest():

    def bagging(self,dataset,sample_num,dataset_size,typ = "None"):

        subdata_list = []
        if typ == "None":
            for datalist in range(sample_num):
                sub = random.sample(dataset,dataset_size)
                # print(len(sub[0]),len(sub))
                # print(sub[0][5])
                subdata_list.append(sub)
            return subdata_list
        else :
            if typ == "time_series":
                pass

    def unique_list(self,list):
        output = []
        for x in list:
            if x not in output:
                output.append(x)
        return output

    def vote(self,predic_label):
        # vote_dict = {}
        # for result in predic_label:
        #     if result not in vote_dict.keys():
        #         vote_dict[result] = 1
        #     else:
        #         vote_dict[result] += 1
        # sort_vote_disc = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
        # return sort_vote_disc[0][0]
        result = -1
        num = -1
        for i in self.unique_list(predic_label):
            # predic_label.count(i)
            if predic_label.count(i) > num:
                num = predic_label.count(i)
                result = i
            else:
                continue
        return result


    def fit(self,data_xy,num = 5,dataset_size = 10000,typ = "None"):
        dataset, featurelist = transfrom_to_train(data_xy)
        del featurelist[len(featurelist) - 1]

        tree = decision_tree()
        self.model_list = []
        bagging_dataset = self.bagging(dataset,num,dataset_size,typ)
        for data in tqdm(bagging_dataset):
            model = tree.CreateDecisionTree(data,featurelist)
            self.model_list.append(model)


    def predict(self,test_data):
        predic_label = []
        predic_result =[]
        predic_dict = []
        test_,attrib = transfrom_to_train(test_data)
        # print(attrib)
        for model in tqdm(self.model_list):
            # print(model.items())

            label = decision_tree().Predict(test_,attrib,model)
            # print(label)
            predic_label.append(label)

        for j in tqdm(range(len(predic_label[0]))):
            for i in range(len(predic_label)):

                predic_dict.append(predic_label[i][j])
            predic_result.append(self.vote(predic_dict))

        return predic_result


def train_x_train_y(data,wanna_test = False,test_num = 0.25):

    data1 = data.iloc[np.random.permutation(len(data))]
    if wanna_test :

        data2 = data1[0:round((1-test_num)*len(data))]
        train_y = data2.iloc[:,-1]
        train_x = data2.drop(str(train_y.name), axis = 1)
        data3 = data1[round((1-test_num)*len(data)):]
        test_y = data3.iloc[:,-1]
        test_x = data3.drop(str(test_y.name), axis = 1)

        return train_x,train_y,test_x,test_y
    else:
        train_y = data1.iloc[:, -1]
        train_x = data1.drop(str(train_y.name), axis = 1)
        return train_x, train_y

def confusion_matrix(result,test_y,if_print = True):

    TP,TN,FP,FN = 0,0,0,0
    test_y = np.asarray(test_y)

    if len(test_y) != len(result):
        print(len(result))
        print(len(test_y))

    for i in range(len(result)):
        if result[i] > 0 :
            if result[i] == test_y[i]:
                TP += 1
            else:
                TN += 1
        else:
            if result[i] == test_y[i]:
                FP += 1
            else:
                FN += 1
    accuracy = (TP + FP) / len(result)
    precision = (TP)/(TP + TN + 0.01)
    recall = TP/(TP + FP + 0.01)
    F1 = 2*(precision*recall)/(precision + recall + 0.01)
    cm = np.array([[TP,FN],[TN,FP]])
    if if_print:
        print("confusion_matrix:")
        print(cm)
        print("accuracy:",accuracy)
        print("precision",precision)
        print("recall",recall)
        print("F1-score",F1)
    return accuracy,precision,recall,F1

#referrance https://www.bilibili.com/video/BV1MK4y1P7TB/

if __name__ == '__main__':
    data3 = pd.read_csv("train.csv")
    data3 = data3.drop(columns="policy_id")
    train_x, train_y, test_x, test_y = train_x_train_y(data3, wanna_test=True)
    train_x_y = pd.concat([train_x, train_y], axis=1)
    # train_,attributeList = transfrom_to_train(data=train_x_y)
    randomforest = random_forest()

    model_forest = randomforest.fit(train_x_y,num=5,dataset_size = 10000)
    result = randomforest.predict(test_x)
    print(result)
    confusion_matrix(result,test_y)



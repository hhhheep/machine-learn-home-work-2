import numpy as np
import pandas as pd
import math
from tqdm import tqdm

class Naive_bayes(object):
    def norm(self,val,mean,std):
        pdf = 1/math.sqrt(2*math.pi*std) * math.exp(math.pow(val-mean,2)/(2*std*std))
        return pdf

    def fit(self,train:pd.DataFrame):
        label = train.columns.values[-1]
        feature = train.columns.values[:-1]
        parameters = {}
        label_value = train[label].unique()
        parameters[label] = {}
        for val in label_value:
            D_C = len(train[train[label] == val])
            D = len(train)
            N = len(label_value)
            parameters[label][val] = (D_C + 1)/N+D

        for fea in tqdm(feature):
            if fea not in parameters.keys():
                parameters[fea] = {}
            # print("int" in str(train[fea].dtype))
            if ("object" in str(train[fea].dtype)) or ("int" in str(train[fea].dtype)):
                feature_value = train[fea].unique()
                N_i = len(feature_value)
                for feature_val in feature_value:
                    parameters[fea][feature_val] = {}
                    for label_val in label_value:
                        # print(fea,feature_val,label_val)
                        D_ci = len(train.loc[(train[label] == label_val) & (train[fea] == feature_val)])
                        D_c = len(train[train[label] == label_val][fea])
                        parameters[fea][feature_val][label_val] = (D_ci+1)/(N_i+D_c)

            else:
                for label_val in label_value:
                    parameters[fea][label_val] = {}
                    mean_f = train[train[label] == label_val][fea].mean()
                    std_f = train[train[label] == label_val][fea].std()
                    parameters[fea][label_val]["mean"] = mean_f
                    parameters[fea][label_val]["std"] = std_f
        self.parameters = parameters
        return self.parameters

    def predict(self,test:pd.DataFrame,label,parameters = "None"):
        if parameters == "None":
            parameters = self.parameters

        predic_label = []
        feature = test.columns.values
        # print(feature)
        label_n = label.columns.values
        label_values = label[str(label_n[0])].unique()
        # print(label_values)
        max_p = -999

        res = ""
        p = 0
        if len(test.index) != len(label):
            print(len(test.index))
            print(len(label))
        for value in tqdm(test.index):
            for label_val in label_values:
                p = parameters[str(label_n[0])][label_val]
                for fea in feature:


                    if ("object" in str(test[fea].dtype)) or ("int" in str(test[fea].dtype)):
                        p *= parameters[fea][test.loc[value,fea]][label_val]
                    else:
                        p *= self.norm(test.loc[value,fea],mean = parameters[fea][label_val]["mean"],std = parameters[fea][label_val]["std"])

                if p > max_p:
                    res = label_val
                    max_p = p
            predic_label.append(res)

        return  predic_label


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
    precision = (TP) / (TP + TN + 0.01)
    recall = TP / (TP + FP + 0.01)
    F1 = 2 * (precision * recall) / (precision + recall + 0.01)
    cm = np.array([[TP,FN],[TN,FP]])
    if if_print:
        print("confusion_matrix:")
        print(cm)
        print("accuracy:",accuracy)
        print("precision",precision)
        print("recall",recall)
        print("F1-score",F1)
    return accuracy,precision,recall,F1

# reference https://blog.csdn.net/CarryLvan/article/details/109236906

if __name__ == '__main__':
    data3 = pd.read_csv("train.csv")
    data3 = data3.drop(columns="policy_id")
    train_x, train_y, test_x, test_y = train_x_train_y(data3, wanna_test=True)
    train_x_y = pd.concat([train_x, train_y], axis=1)
    NB_model = Naive_bayes()
    model_naive = NB_model.fit(train_x_y)

    result = NB_model.predict(pd.DataFrame(test_x),pd.DataFrame(test_y),model_naive)
    # print(result)
    # print(train_y)
    confusion_matrix(result,test_y)











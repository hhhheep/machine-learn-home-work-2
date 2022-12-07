from sklearn.ensemble import RandomForestClassifier
from Naivebeyes import confusion_matrix
import pandas as pd
# from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

data = pd.read_csv("train.csv")
data = data.drop(columns="policy_id")
train_x,train_y,test_x,test_y  = train_x_train_y(data,wanna_test = True,test_num = 0.25)
# train_x,train_y,test_x,test_y = train_x.to_numpy(),train_y.to_numpy(),test_x.to_numpy(),test_y.to_numpy()

# print(train_y.shape)
# train_y = np.reshape(arousal_lable,(624,1))
rfc = RandomForestClassifier(random_state=0,class_weight="balanced")
rfc = rfc.fit(train_x,train_y)
score_r = rfc.score(test_x,test_y)
pred = rfc.predict(test_x)
print(pred)

confusion_matrix(pred,test_y,if_print = True)

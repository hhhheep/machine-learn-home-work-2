import numpy as np
import math
import pandas as pd
from tqdm import tqdm
from Naivebeyes import Naive_bayes,train_x_train_y,confusion_matrix
from randomforest import random_forest
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier,Pool
import xgboost as xgb
import lightgbm as lgb


class K_fold_cross_validation(object):

    def __init__(self,k = 5,dataset = "None",model = "None"):
        """
        :type fit: class : fit
        :type predic: class : predic
        :type dataset: pd.Datafram
        """
        self.k = k
        self.data = dataset
        self.model = model


    def Average(self,lst):
        return sum(lst) / len(lst)

    def cut_data(self):
        df = self.data
        k = self.k
        df_num = len(df)
        df = df.iloc[np.random.permutation(len(df))]
        every_epoch_num = int((df_num / k))
        self.cut_reslut = []
        for index in (range(k)):#tqdm
            if index < k - 1:
                df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
            else:
                df_tem = df[every_epoch_num * index:]
            self.cut_reslut.append(df_tem)
        return self.cut_reslut
        # 原文链接：https: // blog.csdn.net / weixin_42599499 / article / details / 117809308
    def validation(self,test_x,test_y,train_need_y = False,need_label = False,is_xgb = False,is_lg = False,is_cat = False):
        self.cut_data()
        mean_accuracy, mean_precision, mean_recall, mean_F1 = [],[],[],[]
        for item in (range(self.k)):#tqdm
            datalist = self.cut_reslut
            vali_data = datalist[item]

            traindata = pd.DataFrame()
            for index in range(len(datalist)):
                if index == item:
                    continue
                else:
                    traindata = traindata.append(datalist[index])
            # vali_y = vali_data.iloc[:, -1]
            # vali_x = vali_data.drop(str(vali_y.name), axis=1)

            train_y = traindata.iloc[:, -1]
            train_x = traindata.drop(str(train_y.name), axis=1)

            if is_xgb:
                param = {'max_depth': 7, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
                param['nthread'] = 4
                param['seed'] = 100
                param['eval_metric'] = 'auc'
                dtrain = self.model.DMatrix(data=train_x, label=train_y, missing=-999.0)
                dpredict = self.model.DMatrix(test_x)
                # evallist = [(dtest, 'eval'), (dtrain, 'train')]
                xg = self.model.train(param, dtrain, 1000,early_stopping_rounds=10)
                result = xg.predict(dpredict)
            else:
                if is_lg:
                    train_data = self.model.Dataset(data=train_x, label=train_y)
                    test_data = self.model.Dataset(data=test_x, label=test_y)

                    param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
                    param['metric'] = 'rmse'

                    bst = self.model.train(param, train_data, 1000, valid_sets=[test_data])
                    result = bst.predict(test_data, num_iteration=bst.best_iteration)

                else:
                    if is_cat:
                        train_data = Pool(train_x, train_y)
                        self.model.fit(train_data,)
                        result = self.model.predic(test_x)
                    else:
                        if train_need_y:
                            self.model.fit(train_x, train_y)
                        else:
                            self.model.fit(traindata)
                        if need_label:

                            result = self.model.predict(pd.DataFrame(test_x), pd.DataFrame(test_y))
                        else:
                            result = self.model.predict(test_x)

            a1,a2,a3,a4 = confusion_matrix(result,test_y,if_print=False)
            mean_accuracy.append(a1)
            mean_precision.append(a2)
            mean_recall.append(a3)
            mean_F1.append(a4)
        print('mean_accuracy: %s, mean_precision: %s, mean_recall: %s,mean_F1: %s' % (self.Average(mean_accuracy),self.Average(mean_precision),self.Average(mean_recall),self.Average(mean_F1)))
if __name__ == '__main__':
    data3 = pd.read_csv("train.csv")
    data3 = data3.drop(columns="policy_id")
    train_x, train_y, test_x, test_y = train_x_train_y(data3, wanna_test=True)
    train_x_y = pd.concat([train_x, train_y], axis=1)
    for k in [5]:
        # ------------- Naive bayes -------------
        NB = Naive_bayes()
        cv_N = K_fold_cross_validation(k=k, dataset = train_x_y,model=NB)
        cv_N.validation(test_x=test_x,test_y=test_y,need_label=True)
        #------------- random forest -------------
        rfc = random_forest()
        cv_R = K_fold_cross_validation(k=k, dataset = train_x_y,model=rfc)
        cv_R.validation(test_x=test_x,test_y=test_y)
        #------------- sklearn random forest-------------
        srfc = RandomForestClassifier(random_state=0, class_weight="balanced")
        cv_SR = K_fold_cross_validation(k=k, dataset=train_x_y, model=srfc)
        cv_SR.validation(test_x=test_x,test_y=test_y,train_need_y = True)
        # ------------- XGboost -------------
        # model_XG =
        cv_XG = K_fold_cross_validation(k=k, dataset=train_x_y, model=xgb)
        cv_XG.validation(test_x=test_x,test_y=test_y,is_xgb=True)
        # ------------- catboost -------------
        model_cat = CatBoostClassifier(iterations=10, learning_rate=1, depth=2)
        cv_CAT = K_fold_cross_validation(k=k, dataset=train_x_y, model=model_cat)
        cv_CAT.validation(test_x=test_x,test_y=test_y,is_cat=True)
        #------------- LightGBM -------------
        # model_LG = lgb()
        cv_LG = K_fold_cross_validation(k=k, dataset=train_x_y, model=lgb)
        cv_XG.validation(test_x=test_x,test_y=test_y,is_lg=True)


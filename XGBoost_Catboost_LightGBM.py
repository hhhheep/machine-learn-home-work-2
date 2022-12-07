#——————————————————— xgboost ———————————————————

import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import matplotlib
from Naivebeyes import confusion_matrix
from sklearn.metrics import mean_squared_error


data = pd.read_csv("train.csv")
data = data.drop(columns="policy_id")

y = data.iloc[:,-1]
X = data.drop(str(y.name), axis = 1)

x_train, x_predict, y_train, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=100)


dtrain = xgb.DMatrix(data=x_train,label=y_train,missing=-999.0)
dtest = xgb.DMatrix(data=x_test,label=y_test,missing=-999.0)

param = {'max_depth': 7, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['seed'] = 100
param['eval_metric'] = 'auc'


num_round = 10
evallist = [(dtest, 'eval'), (dtrain, 'train')]


bst_with_evallist_and_early_stopping_10 = xgb.train(param, dtrain, num_round*100, evallist,early_stopping_rounds=10)
dpredict = xgb.DMatrix(x_predict)
ypred_with_evallist_and_early_stopping_100 = bst_with_evallist_and_early_stopping_10.predict(dpredict)
ypred_with_evallist_and_early_stopping_100 = [round(i) for i in ypred_with_evallist_and_early_stopping_100]
confusion_matrix(ypred_with_evallist_and_early_stopping_100,y_predict)
# print(ypred_with_evallist_and_early_stopping_100)

# print("RMSE of bst_with_evallist_and_early_stopping_100 ：", np.sqrt(mean_squared_error(y_true=y_predict,y_pred=ypred_with_evallist_and_early_stopping_100)))


# —————————————————— catboost ——————————————————————


from catboost import CatBoostClassifier,Pool

cat_features = [0, 1]  # 类别特征下标

x_train, x_predict, y_train, y_predict = train_test_split(X, y, test_size=0.10, random_state=100)
# print(x_train.shape)
# print(y_train.shape)
# 定义模型
train_data = Pool(x_train,y_train)
model = CatBoostClassifier(iterations=10, learning_rate=1, depth=2)

# 训练
model.fit(train_data)

# 预测类别
preds_class = model.predict(x_predict)
print(preds_class)
confusion_matrix(preds_class,y_predict)
# —————————————————— LightGBM ——————————————————————

import lightgbm as lgb


train_data = lgb.Dataset(data=x_train,label=y_train)
test_data = lgb.Dataset(data=x_test,label=y_test)

param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}
param['metric'] = 'rmse'

bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
ypred = bst.predict(x_predict, num_iteration=bst.best_iteration)
ypred = [round(i) for i in ypred]
print(ypred)
confusion_matrix(ypred,y_predict)
from sklearn.metrics import mean_squared_error
RMSE = np.sqrt(mean_squared_error(y_predict, ypred))
print("RMSE of predict :",RMSE)


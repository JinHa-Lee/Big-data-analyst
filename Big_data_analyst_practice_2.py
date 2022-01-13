# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler , LabelEncoder, MinMaxScaler , RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from itertools import cycle


x = pd.read_csv("./X_train.csv",encoding='cp949')
y = pd.read_csv("./y_train.csv")
x_1 = pd.read_csv("./X_test.csv",encoding='cp949')


x.fillna(0,inplace=True)
x_1.fillna(0,inplace=True)


print(set(x['주구매상품'])-set(x_1['주구매상품']))
print(set(x['주구매지점'])-set(x_1['주구매지점']))

print(set(x_1['주구매상품'])-set(x['주구매상품']))
print(set(x_1['주구매지점'])-set(x['주구매지점']))

x[x['주구매상품']=='소형가전']

pd.set_option('display.max_columns',100)
x = x[x['주구매상품']!='소형가전'].reset_index(drop=True)
y = y[(y['cust_id']!=1521)&(y['cust_id']!=2035)].reset_index(drop=True)

x_num = x.drop(['cust_id','주구매상품','주구매지점'],axis=1)
x_1_num = x_1.drop(['cust_id','주구매상품','주구매지점'],axis=1)
columns_index = x_num.columns                            
x_str = x[['주구매상품','주구매지점']]
x_1_str = x_1[['주구매상품','주구매지점']]

# le1 = LabelEncoder()
# le1.fit(x['주구매상품'])
# x_str['주구매상품'] = le1.transform(x_str['주구매상품'])
# x_1_str['주구매상품'] = le1.transform(x_1_str['주구매상품'])
# le2 = LabelEncoder()
# le2.fit(x['주구매지점'])
# x_str['주구매지점'] = le2.transform(x_str['주구매지점'])
# x_1_str['주구매지점'] = le2.transform(x_1_str['주구매지점'])

x_str = pd.get_dummies(x_str)
x_1_str = pd.get_dummies(x_1_str)

# scaler = StandardScaler()
# scaler.fit(x_num)
# x_num = scaler.transform(x_num)
# x_1_num = scaler.transform(x_1_num)
# x_num = pd.DataFrame(x_num,columns=columns_index)
# x_1_num = pd.DataFrame(x_1_num,columns=columns_index)


scaler2 = RobustScaler()
scaler2.fit(x_num)
x_num = scaler2.transform(x_num)
x_1_num = scaler2.transform(x_1_num)
x_num = pd.DataFrame(x_num,columns=columns_index)
x_1_num = pd.DataFrame(x_1_num,columns=columns_index)

y.drop(['cust_id'],axis=1,inplace=True)

x_dum = pd.concat([x_num,x_str],axis=1)
x_1_dum = pd.concat([x_1_num,x_1_str],axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(x_dum,y,test_size=0.3 , random_state = 8)
fold1 = StratifiedKFold(n_splits=10)



dt = DecisionTreeClassifier()
params = {
    'max_depth' : [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
grid_dt = GridSearchCV(dt, param_grid=params, scoring='accuracy', cv=fold1)
grid_dt.fit(X_train, Y_train)
tree_best = grid_dt.best_estimator_
proba_dt = tree_best.predict_proba(X_test)[:,1]

# dt.fit(X_train,Y_train)
# proba_dt = dt.predict_proba(X_test)[:,1]

result_dt = roc_auc_score(Y_test, proba_dt)


lg = LogisticRegression()
params = {
    'C' : [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
}
grid_lg = GridSearchCV(lg, param_grid=params, scoring='accuracy', cv=fold1)
grid_lg.fit(X_train, Y_train)
logi_best = grid_lg.best_estimator_
proba_lg = logi_best.predict_proba(X_test)[:,1]

# lg.fit(X_train,Y_train)
# proba_lg = lg.predict_proba(X_test)[:,1]

result_lg = roc_auc_score(Y_test, proba_lg)


kn = KNeighborsClassifier()
params = {
    'n_neighbors':[1,2,3,4,5,6,7,8,9,10]
}
grid_kn = GridSearchCV(kn, param_grid=params, scoring='accuracy', cv=fold1)
grid_kn.fit(X_train, Y_train)
knn_best = grid_kn.best_estimator_
proba_kn = knn_best.predict_proba(X_test)[:,1]

# kn.fit(X_train,Y_train)
# proba_kn = kn.predict_proba(X_test)[:,1]

result_kn = roc_auc_score(Y_test, proba_kn)


vot = VotingClassifier(estimators=[('DT',tree_best),('LR',logi_best),('knn',knn_best)],voting='soft')
vot.fit(X_train,Y_train)
proba_vot = vot.predict_proba(X_test)[:,1]
result_vot = roc_auc_score(Y_test, proba_vot)


rf = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 999)
rf.fit(X_train,Y_train)
proba_rf = rf.predict_proba(X_test)[:,1]
result_rf = roc_auc_score(Y_test, proba_rf)

xg = XGBClassifier()
xg.fit(X_train,Y_train)
proba_xg = xg.predict_proba(X_test)[:,1]
result_xg = roc_auc_score(Y_test,proba_xg)

proba_list = [proba_dt,proba_lg,proba_kn,proba_vot,proba_rf,proba_xg]
proba_list_name = ['tree','logi','knn','vot','rf','xg']
classification_list = [dt,lg,kn,vot,rf,xg]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(len(proba_list)):
    fpr[i], tpr[i], _ = roc_curve(Y_test, proba_list[i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure()
lw = 2
colors = cycle(["aqua", "darkorange", "cornflowerblue","gray","gold","red"])
for i, color in zip(range(len(proba_list)), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of {0} (area = {1:0.2f})".format(proba_list_name[i], roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()


max_result = max(roc_auc, key = roc_auc.get)

prob_fin = classification_list[max_result].predict_proba(x_1_dum)[:,1]
# print(prob_fin)
result_fin = pd.concat([x_1.cust_id,pd.DataFrame(prob_fin)],axis=1)
result_fin.columns = ['cust_id','gender']
# print(result_fin)
result_fin.to_csv('./result.csv')







# params ={
#     'n_estimators':[10,50,100,150,200],
#     'max_depth':[5,6,7,8,9,10],
#     'min_samples_leaf':[8,12,16,20],
#     'min_samples_split':[20,25,30,35,40]
# }

# classifier = RandomForestClassifier()
# grid_rf = GridSearchCV(rf,param_grid=params,cv=fold1)
# grid_rf.fit(X_train,Y_train)
# rf_best = grid_rf.best_estimator_
# proba_rf = rf_best.predict_proba(X_test)[:,1]

# x['내점일수'] = np.log(x['내점일수']+1)
# x_1['내점일수'] = np.log(x_1['내점일수']+1)
# x['내점당구매건수'] = np.log(x['내점당구매건수']+1)
# x_1['내점당구매건수'] = np.log(x_1['내점당구매건수']+1)
# x['구매주기'] = np.log(x['구매주기']+1)
# x_1['구매주기'] = np.log(x_1['구매주기']+1)

# count,bin_dividers = np.histogram(x['내점일수'],bins = 3)
# count2,bin_dividers2 = np.histogram(x['내점당구매건수'],bins = 3)
# count3,bin_dividers3 = np.histogram(x['구매주기'],bins = 3)
# x['내점일수'] = pd.cut(x['내점일수'],bins=bin_dividers,include_lowest=True)
# x['내점당구매건수'] = pd.cut(x['내점당구매건수'],bins=bin_dividers2,include_lowest=True)
# x['구매주기'] = pd.cut(x['구매주기'],bins=bin_dividers3,include_lowest=True)
# x_1['내점일수'] = pd.cut(x_1['내점일수'],bins=bin_dividers,include_lowest=True)
# x_1['내점당구매건수'] = pd.cut(x_1['내점당구매건수'],bins=bin_dividers2,include_lowest=True)
# x_1['구매주기'] = pd.cut(x_1['구매주기'],bins=bin_dividers3,include_lowest=True)
# x_dum = pd.get_dummies(x)
# x_1_dum = pd.get_dummies(x_1)


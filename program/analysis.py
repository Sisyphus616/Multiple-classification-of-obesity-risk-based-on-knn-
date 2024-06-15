import pandas as pd
import numpy as np
from knn import knn
from sklearn.model_selection import train_test_split

# 读入数据
train = pd.read_csv("train.csv")

x_train = []
y_train = []

for i in range(len(train)):
    temp = []
    temp.append(train["Height"][i])
    temp.append(train["Weight"][i])
    x_train.append(temp)

for tp in train['NObeyesdad']:
    if 'Insufficient' in tp:
        y_train.append(0)
    elif 'Normal' in tp:
        y_train.append(1)
    elif 'Type_III' in tp:
        y_train.append(2)
    elif 'Type_II' in tp:
        y_train.append(3)
    elif 'Type_I' in tp:
        y_train.append(4)
    elif 'Level_II' in tp:
        y_train.append(5)
    else:
        y_train.append(6)

x_train = np.array(x_train)
x_train,x_test,y_train,y_label = train_test_split(x_train,y_train)

y_test = []

for x in x_test:
    tp = knn(x,x_train,y_train,k=10)
    y_test.append(tp)

total = 0;cnt = 0

for i in range(len(y_test)):
    if y_label[i]>=2:
        total += 1
    if y_test[i]>=2 and y_label[i]>=2:
        cnt += 1

recall = cnt*1.0/total

print("召回率为：",recall)   # 0.9791777

cnt = 0;total = 0

for i in range(len(y_test)):
    if y_test[i]>=2:
        total += 1
    if y_test[i]>=2 and y_label[i]>=2:
        cnt += 1

precision = cnt*1.0/total
print("精确率为：",precision)

print("F1值为：",2*recall*precision/(recall+precision))
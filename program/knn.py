import pandas as pd
import numpy as np
# import sklearn.preprocessing

# 读入数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x_train = []
y_train = []

for i in range(len(train)):
    temp = []
    temp.append(train["Height"][i])
    temp.append(train["Weight"][i])
    x_train.append(temp)

x_train = np.array(x_train)
# x_train = sklearn.preprocessing.scale(x_train)

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

x_test = []

for i in range(len(test)):
    temp = []
    temp.append(test["Height"][i])
    temp.append(test["Weight"][i])
    x_test.append(temp)

x_test = np.array(x_test)
# x_test = sklearn.preprocessing.scale(x_test)

# K近邻算法函数
def knn(inX,dataSet,labels,k=3):
    # 计算距离
    n = np.size(dataSet,0)
    dist = np.ones(n)
    for i in range(0,n,1):
        disx = dataSet[i][0]-inX[0]
        disy = dataSet[i][1]-inX[1]
        dis = ((disx**2)+(disy**2))**0.5
        dist[i] = dis
    # 从小到大排序
    sort_dist=dist.argsort()
    # 计数
    Count={}
    # 获得前k个近邻点的类型
    for i in range(k):
        Label_flag = labels[sort_dist[i]]
        Count[Label_flag]=Count.get(Label_flag,0)+1
    
    # 初始化
    res=0
    cnt=-1
    for key,value in Count.items():
        if value > cnt:
            res = key
            cnt = value
 
    return res

y_test = []

for x in x_test:
    tp = knn(x,x_train,y_train,k=10)
    if tp==0:
        y_test.append("Insufficient_Weight")
    elif tp==1:
        y_test.append("Normal_Weight")
    elif tp==2:
        y_test.append("Obesity_Type_III")
    elif tp==3:
        y_test.append("Obesity_Type_II")
    elif tp==4:
        y_test.append("Obesity_Type_I")
    elif tp==5:
        y_test.append("Overweight_Level_II")
    else:
        y_test.append("Overweight_Level_I")

output = pd.DataFrame(data={"id":test["id"],"sentiment":y_test})
output.to_csv("./result.csv",index=False)
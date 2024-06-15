import pandas as pd
import numpy as np

# 读入数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x_train = []
y_train = []

for i in range(len(train)):
    temp = []
    temp.append(1)
    temp.append(train["Height"][i])
    temp.append(train["Weight"][i])
    x_train.append(temp)

x_train = np.array(x_train)

for x in x_train: 
    x[1] /= 1.5
    x[2] /= 80
# print(x_train)

for tp in train['NObeyesdad']:
    if 'Insufficient' in tp:
        y_train.append(0)
    elif 'Normal' in tp:
        y_train.append(1)
    elif 'Type_III' in tp:
        y_train.append(4)
    elif 'Type_II' in tp:
        y_train.append(3)
    elif 'Type_I' in tp:
        y_train.append(2)
    elif 'Level_II' in tp:
        y_train.append(6)
    else:
        y_train.append(5)

x_test = []

for i in range(len(test)):
    temp = []
    temp.append(1)
    temp.append(test["Height"][i])
    temp.append(test["Weight"][i])
    x_test.append(temp)

x_test = np.array(x_test)

for x in x_test:
    x[1] /= 1.5
    x[2] /= 80

def sigmoid(x):
    return 6/(1+np.exp(-x))

def fit(x,y,eta=0.2,n_iters=1000):
    thetas = np.ones(x.shape[1])
    a = np.ones(len(x))
    for i in range(n_iters):
        for j in range(len(x)):
            a[j] = sigmoid(np.dot(x[j],thetas))
            if abs(a[j]-y[j])<0.01:
                break
            thetas = thetas - eta*(a[j]-y[j])*x[j]
    return thetas

res = fit(x_train,y_train)
dat = []
y_test = []
for x in x_test:
    dat.append(np.dot(x,res))
for tp in dat:
    if tp<0.5:
        y_test.append("Insufficient_Weight")
    elif tp>=0.5 and tp<1.5:
        y_test.append("Normal_Weight")
    elif tp>=1.5 and tp<2.5:
        y_test.append("Obesity_Type_I")
    elif tp>=2.5 and tp<3.5:
        y_test.append("Obesity_Type_II")
    elif tp>=3.5 and tp<4.5:
        y_test.append("Obesity_Type_III")
    elif tp>=4.5 and tp<5.5:
        y_test.append("Overweight_Level_I")
    else:
        y_test.append("Overweight_Level_II")

output = pd.DataFrame(data={"id":test["id"],"sentiment":y_test})
output.to_csv("./result_Logi.csv",index=False)
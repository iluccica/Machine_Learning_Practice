import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import  StandardScaler

data = pd.read_csv('pneumonia.csv')

#数据归一
# ss = StandardScaler()
# scale_features = data.columns.values
# data[scale_features] = ss.fit_transform(data[scale_features])
# print(data)

##统计信息,均值 标准差
#print(data.describe())

##找到缺失值
#print(data[data.isnull()==True].count())

#特征箱型图
# data.boxplot()
# plt.show()
# plt.savefig("boxplot.jpg")

#相关系数矩阵
# print(data.corr())

#单个变量和结果的回归图
# sns.pairplot(data,x_vars=['days','Suspected cases(1 day before)','close exposure(1 day before)'],y_vars='increment',
#              size=4,kind='reg')
# plt.show()

#建立模型
epoch = 10
model = LinearRegression()
w_total = np.empty([10,3])
for i in range(0,epoch):
    model.fit(data[np.delete(data.columns,1)],data['increment'])
    b = round(model.intercept_,3)
    np.set_printoptions(precision=3, suppress=True)
    w = model.coef_
    # print("b = ",b,"\nw = ",w)
    w_total[i] = w

#权重增加列名
w_total = pd.DataFrame(w_total,columns=['days','Suspected cases(1 day before)','close exposure(1 day before)'])

#这个例子里十次的权重都是一样的,随便拿一组
new_w = w_total.loc[0]

#计算某个特征对结果的影响
effect = data[np.delete(data.columns,1)]
for idx in np.delete(data.columns,1):
    effect[idx] = effect[idx]*new_w[idx]

#影响图
effect.boxplot(vert = False)
plt.show()

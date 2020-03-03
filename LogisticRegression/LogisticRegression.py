import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LogisticRegression


#读数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#先把两个集合并在一起处理数据
train = pd.concat([train,test],keys=(['train','test']))
# print(train.info())
# print(train.isnull().sum())

# c = train.Cabin.value_counts()
# print(c)

# cabin类太多,扔了
train = train.drop(labels='Cabin', axis = 1)


#Embarked用频率最多的S填充
train.Embarked=train.Embarked.fillna('S')


#姓名与年龄相关,做均值分类
train['cc'] = train['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
c = train.loc[:,['cc','Age']].query('Age>0').groupby('cc').mean()

print(c)

#填入年龄
train['Age'] = train['Age'].fillna(0)

value = ['Mr','Miss','Mrs','Master','Dr']
for v in value:
    train.loc[(train.Age==0)&(train.cc==v),'Age'] = c.loc[v,'Age']

#处理分类特种 赋值
categoricial = train.dtypes[train.dtypes=='object'].index
print(categoricial)

train = train.replace({'Sex':{'male':1,'female':2},
                       'Embarked':{'S':1,'C':2,'Q':3}})

#去掉没用的

train = train.drop(labels=['cc','Name','Ticket'],axis=1)

#分开数据
train_data = train.xs('train')
test_data = train.xs('test').drop(labels = 'Survived',axis = 1)
X_train = train_data.drop(labels = 'Survived',axis = 1)
y_train = train_data['Survived']
test_data = test_data.fillna(0)

#~~~~~~~~~~~~~~~~~~数据处理over~~~~~~建立模型~~~~~~~~~~~~~~~~~

S = StandardScaler()
# print (X_train)
S.fit(X_train)
X_train_stand = S.transform(X_train)
X_test_stand = S.transform(test_data)
Log = LogisticRegression(C=10)
Log.fit(X_train_stand,y_train)
prediction = Log.predict(X_test_stand)
result = pd.DataFrame({'PassengerId':test_data.index,'Survived':prediction.astype(np.int32)})
print(test_data)
print(Log.coef_)#查看权重
# result.to_csv('result.csv',index=False)
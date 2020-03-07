import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import  RandomForestRegressor

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





#随机森林年龄回归
rf_train = train.xs('train')
age_df = rf_train[['Age','Fare','Parch','SibSp','Pclass']]

known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()


target_age = known_age[:,0]#年龄回归样本
train_age = known_age[:,1:]


rf_age = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rf_age.fit(train_age,target_age)#训练年龄模型


predicted_Ages = rf_age.predict(unknown_age[:,1:])#预测,后面再用


#处理分类特征 赋值



train = train.replace({'Sex':{'male':1,'female':2},
                       'Embarked':{'S':1,'C':2,'Q':3}})
print(train)
#去掉没用的

train = train.drop(labels=['Name','Ticket'],axis=1)

#分开数据
train_data = train.xs('train')
train_data.loc[(train_data.Age.isnull()),'Age'] = predicted_Ages
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
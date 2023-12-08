import pandas as pd

# 数据列名：PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 读取数据
train_data = pd.read_csv('./data_fixed/train.csv', encoding='utf-8')
test_data = pd.read_csv('./data_fixed/test.csv', encoding='utf-8')

# 数据集划分
X_train = train_data.drop(['PassengerId', 'Survived'], axis=1)
Y_train = train_data['Survived']
X_test = test_data.drop(['PassengerID'], axis=1)    # axis=1 表示删除列
Y_test = test_data['Survived']

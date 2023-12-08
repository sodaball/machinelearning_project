import pandas as pd

# 数据列名：PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 读取数据
train_data = pd.read_csv('./data/train.csv', encoding='utf-8')
test_data = pd.read_csv('./data/test.csv', encoding='utf-8')


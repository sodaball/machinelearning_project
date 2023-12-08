import pandas as pd

# 数据列名：PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 读取已经完成缺失值处理、编码、归一化的数据
train_data = pd.read_csv('./data_fixed/train.csv', encoding='utf-8')
test_data = pd.read_csv('./data_fixed/test.csv', encoding='utf-8')

# 数据集划分
X_train = train_data.drop(['PassengerId', 'Survived'], axis=1)
Y_train = train_data['Survived']
X_test = test_data.drop(['PassengerId'], axis=1)    # axis=1 表示删除列
# Y_test = test_data['Survived']    # 测试集没有标签

# 使用逻辑回归模型进行训练
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, Y_train)

# 使用训练好的模型进行预测
Y_pred = lr.predict(X_test)

# 将预测结果合并到测试集中
test_data['Survived'] = Y_pred

# 打印前十个数据和预测值
print(test_data.head(10))

# 保存预测结果
test_data.to_csv('prediction.csv', index=False)

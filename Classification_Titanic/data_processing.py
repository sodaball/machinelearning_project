import pandas as pd

# 数据列名：PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 读取数据
train_data = pd.read_csv('./data/train.csv', encoding='utf-8')
test_data = pd.read_csv('./data/test.csv', encoding='utf-8')

# 查看是否有缺失值
print(train_data.info())
print('-' * 30)
print(test_data.info())

# 预测 Age 列的缺失值
def age_predict(df):
    # 将 Age 不缺失的项作为训练集，Age 缺失的项作为测试集
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df.loc[(df['Age'].notnull())]
    age_df_isnull = age_df.loc[(df['Age'].isnull())]
    # print(age_df_notnull)
    # print(age_df_isnull)

    # 将训练集和测试集分为特征和标签
    X = age_df_notnull.values[:, 1:]
    y = age_df_notnull.values[:, 0]

    # 使用随机森林回归模型进行训练
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    rfr.fit(X, y)

    # 使用训练好的模型进行预测
    predictedAges = rfr.predict(age_df_isnull.values[:, 1:])
    df.loc[(df['Age'].isnull()), 'Age'] = predictedAges

    return df

# 处理 Cabin 列的缺失值，将缺失和非缺失分为两个类别
train_data['Cabin'] = train_data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
test_data['Cabin'] = test_data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

# 处理 Embarked 列的缺失值，用众数填充
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# 处理 Fare 列的缺失值，用均值填充
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)

# 处理 Age 列的缺失值，使用其他列的数据预测 Age 列的缺失值
# train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
# test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
train_data = age_predict(train_data)
test_data = age_predict(test_data)

# 处理缺失值后再查看是否有缺失值
print(train_data.info())
print('-' * 30)
print(test_data.info())
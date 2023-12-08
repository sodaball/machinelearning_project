import pandas as pd

# 数据列名：PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# 读取数据
train_data = pd.read_csv('./data/train.csv', encoding='utf-8')
test_data = pd.read_csv('./data/test.csv', encoding='utf-8')

# 打印数据
print(train_data.head(10))

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
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# 处理 Fare 列的缺失值，用均值填充
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace=True)
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

# 数据编码
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
train_data['Pclass'] = labelEncoder.fit_transform(train_data['Pclass'])
train_data['Name'] = labelEncoder.fit_transform(train_data['Name'])
train_data['Sex'] = labelEncoder.fit_transform(train_data['Sex'])
train_data['Ticket'] = labelEncoder.fit_transform(train_data['Ticket'])
train_data['Cabin'] = labelEncoder.fit_transform(train_data['Cabin'])
train_data['Embarked'] = labelEncoder.fit_transform(train_data['Embarked'])

test_data['Pclass'] = labelEncoder.fit_transform(test_data['Pclass'])
test_data['Name'] = labelEncoder.fit_transform(test_data['Name'])
test_data['Sex'] = labelEncoder.fit_transform(test_data['Sex'])
test_data['Ticket'] = labelEncoder.fit_transform(test_data['Ticket'])
test_data['Cabin'] = labelEncoder.fit_transform(test_data['Cabin'])
test_data['Embarked'] = labelEncoder.fit_transform(test_data['Embarked'])

# 打印编码后的数据
print("\nafter encoding:")
print(train_data.head(10))

# 数据归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# 对 Age 和 Fare 列进行归一化，因为这两列的数值范围较大
train_data['Age'] = scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))
train_data['Fare'] = scaler.fit_transform(train_data['Fare'].values.reshape(-1, 1))
test_data['Age'] = scaler.fit_transform(test_data['Age'].values.reshape(-1, 1))
test_data['Fare'] = scaler.fit_transform(test_data['Fare'].values.reshape(-1, 1))

# 打印归一化后的数据
print("\nafter normalization:")
print(train_data.head(10))

# 保存处理后的数据
# 创建文件夹
import os
if not os.path.exists('./data_fixed'):
    os.mkdir('./data_fixed')
# 保存文件
train_data.to_csv('./data_fixed/train.csv', index=False)
test_data.to_csv('./data_fixed/test.csv', index=False)

'''
1. **PassengerId（乘客ID）:** 乘客在数据集中的唯一标识。
2. **Survived（是否幸存）:** 目标变量，表示乘客是否在泰坦尼克号沉没事件中幸存。1表示幸存，0表示未幸存。
3. **Pclass（舱位等级）:** 乘客所在的舱位等级，有三个值：1st（上层）、2nd（中层）、3rd（下层）。
4. **Name（姓名）:** 乘客的姓名。
5. **Sex（性别）:** 乘客的性别，值为 "male"（男性）或 "female"（女性）。
6. **Age（年龄）:** 乘客的年龄。可能包含缺失值。
7. **SibSp（同辈亲属数量）:** 乘客在船上有多少兄弟姐妹或配偶。
8. **Parch（非同辈亲属数量）:** 乘客在船上有多少父母或子女。
9. **Ticket（船票号码）:** 乘客的船票号码。
10. **Fare（票价）:** 乘客支付的船票费用。
11. **Cabin（客舱号码）:** 乘客的客舱号码。可能包含缺失值。
12. **Embarked（登船港口）:** 乘客登船的港口，有三个值：C（瑟堡）、Q（皇后镇）、S（南安普顿）。可能包含缺失值。
'''
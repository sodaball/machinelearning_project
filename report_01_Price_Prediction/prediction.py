from sklearn import preprocessing
import pandas as pd

train_data = pd.read_csv('train_data.csv', encoding='utf-8')
test_data = pd.read_csv('test_data.csv', encoding='utf-8')

# 打印数据的前5行
print("train_data:")
print(train_data.head(5))

label_encoder = preprocessing.LabelEncoder()

# 属性包括data共12列
# Direction,District,Elevator,Floor,Garden,Id,Layout,Price,Region,Renovation,Size,Year

# 对Direction,District,Elevator,Garden,Layout,Region,Renovation进行编码
# 由于这些属性的值都是离散的，所以采用LabelEncoder进行编码
train_data['Direction'] = label_encoder.fit_transform(train_data['Direction'])
train_data['District'] = label_encoder.fit_transform(train_data['District'])
train_data['Elevator'] = label_encoder.fit_transform(train_data['Elevator'])
train_data['Garden'] = label_encoder.fit_transform(train_data['Garden'])
train_data['Layout'] = label_encoder.fit_transform(train_data['Layout'])
train_data['Region'] = label_encoder.fit_transform(train_data['Region'])
train_data['Renovation'] = label_encoder.fit_transform(train_data['Renovation'])

test_data['Direction'] = label_encoder.fit_transform(test_data['Direction'])
test_data['District'] = label_encoder.fit_transform(test_data['District'])
test_data['Elevator'] = label_encoder.fit_transform(test_data['Elevator'])
test_data['Garden'] = label_encoder.fit_transform(test_data['Garden'])
test_data['Layout'] = label_encoder.fit_transform(test_data['Layout'])
test_data['Region'] = label_encoder.fit_transform(test_data['Region'])
test_data['Renovation'] = label_encoder.fit_transform(test_data['Renovation'])

# 打印编码后的数据的前5行
print("\ntrain_data_encoded:")
print(train_data.head(5))

# 由于Size和Year的值是连续的，所以采用MinMaxScaler进行归一化
min_max_scaler = preprocessing.MinMaxScaler()
train_data['Size'] = min_max_scaler.fit_transform(train_data['Size'].values.reshape(-1, 1))
train_data['Year'] = min_max_scaler.fit_transform(train_data['Year'].values.reshape(-1, 1))

test_data['Size'] = min_max_scaler.fit_transform(test_data['Size'].values.reshape(-1, 1))
test_data['Year'] = min_max_scaler.fit_transform(test_data['Year'].values.reshape(-1, 1))

# 打印归一化后的数据的前5行
print("\ntrain_data_normalized:")
print(train_data.head(5))

# 划分训练集和测试集的属性和标签
X_train = train_data[['Direction', 'District', 'Elevator', 'Floor', 'Garden', 'Layout', 'Region', 'Renovation', 'Size', 'Year']]
Y_train = train_data['Price']
X_test = test_data[['Direction', 'District', 'Elevator', 'Floor', 'Garden', 'Layout', 'Region', 'Renovation', 'Size', 'Year']]
Y_test = test_data['Price']

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 随机森林回归模型
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, Y_train)

# 打印回归的评价指标，包括平均绝对误差、均方误差、R2得分、解释方差得分
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
Y_pred = rf.predict(X_test)
print("\nMean Absolute Error: ", mean_absolute_error(Y_test, Y_pred))
print("Mean Squared Error: ", mean_squared_error(Y_test, Y_pred))
print("R2 Score: ", r2_score(Y_test, Y_pred))
print("Explained Variance Score: ", explained_variance_score(Y_test, Y_pred))

# 随机森林回归模型，并使用GridSearchCV选择最优参数
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
param_grid = {'n_estimators': range(10, 100, 10), 'max_depth': range(3, 14, 2), 'min_samples_split': range(50, 201, 20)}
grid = GridSearchCV(rf, param_grid, cv=5)   # 这一步的作用是选择最优参数，但是这里的cv=5是5折交叉验证
grid.fit(X_train, Y_train)  # q:训练的是回归模型
print(grid.best_params_)    # 打印最优参数

# 打印回归的评价指标，包括平均绝对误差、均方误差、R2得分、解释方差得分
Y_pred = grid.predict(X_test)
print("\nafter GridSearchCV:")
print("Mean Absolute Error: ", mean_absolute_error(Y_test, Y_pred))
print("Mean Squared Error: ", mean_squared_error(Y_test, Y_pred))
print("R2 Score: ", r2_score(Y_test, Y_pred))
print("Explained Variance Score: ", explained_variance_score(Y_test, Y_pred))

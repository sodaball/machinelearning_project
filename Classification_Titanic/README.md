# Report 2 - Titanic

[RMS泰坦尼克号](https://en.wikipedia.org/wiki/RMS_Titanic)的沉没是历史上最臭名昭著的海难之一。完成对**哪些人更有可能生存的分析**，本质是二分类问题，难点在于数据缺失值的处理。

### Ⅰ. 数据

数据分为：

* 训练数据：`train.csv`，共有891条数据
* 测试数据：`test.csv`，共有418条数据



数据文件为`csv`格式的文本文件，可以使用 `pandas`库读取，具体的数据格式如下图所列：

![data description1](images/data_description1.png)

train数据共有11列

test数据共有10列， 比train少了survival列



### Ⅱ. 步骤

#### 1. 数据读取

使用pandas

#### 2. 数据校验

查看是否有缺失值、异常值，并进行填充、修正

```python
print(train_data.info())
print('-' * 30)
print(test_data.info())
```



#### 2. 数据编码

针对非数值型数据。

* 映射为唯一的整数。可以使用`sklearn.processing.LabelEncoder()`,每个不同的类别都被映射为一个唯一的整数值。但是，`LabelEncoder` 会在对具有大小关系的类别进行编码时引入错误的大小关系

* 映射为独热码。若某些非数值数据具有大小关系，如"small"、"big"，此时应该使用One-Hot Encoding



#### 3. 归一化(可选)

（针对连续型的数值型数据），可以使用`sklearn.processing.MinMaxScaler`



处理前后的数据：

![data_processing](Q:\Dauhau_data_学习资料\互助群\机器学习\machinelearning_homework\Regression_House_Price_Prediction\images\data_processing.png)



#### 4. 划分数据集

```python
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.1, random_state=0)
```



#### 5. 训练，拟合

比如使用随机森林回归模型，可以使用GridSearchCV选择最优参数

搜索过程比较慢，可以调整搜索的范围和步长，还可以并行搜索，添加参数`n_jobs=8`

```python
grid = GridSearchCV(rf, param_grid, cv=2, verbose=2, n_jobs=<num_threads>)   # 这一步的作用是选择最优参数, 但是这里的cv=3是3折交叉验证, n_jobs是并行数
```

最佳参数：

![params](Q:\Dauhau_data_学习资料\互助群\机器学习\machinelearning_homework\Regression_House_Price_Prediction\images\params.png)





#### 6. 打印评价指标

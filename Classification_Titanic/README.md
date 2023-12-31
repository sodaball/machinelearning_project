# Report 2 - Titanic

[RMS泰坦尼克号](https://en.wikipedia.org/wiki/RMS_Titanic)的沉没是历史上最臭名昭著的海难之一。完成对**哪些人更有可能生存的分析**，本质是二分类问题，难点在于数据缺失值的处理。

### Ⅰ. 数据

数据分为：

* 训练数据：`train.csv`，共有891条数据
* 测试数据：`test.csv`，共有418条数据



数据文件为`csv`格式的文本文件，可以使用 `pandas`库读取，具体的数据格式如下图所列：

<img src="images/data_description.png" alt="data description" style="zoom:67%;" />

train数据共有12列

test数据共有11列， 比train少了survival列

每列含义：

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



### Ⅱ. 步骤

#### 1. 数据读取

.npy文件，用`np.load()`读取

.csv文件，用`pd.read_csv`读取

#### 2. 数据校验

查看是否有缺失值、异常值，并进行填充、修正

```python
print(train_data.info())
print('-' * 30)
print(test_data.info())
```

<img src="images/data_missing.png" alt="data_missing" style="zoom:50%;" />

可见train数据集中Age列和Cabin列大量缺失，Embarked少量缺失

test数据集中Cabin列大量缺失，Fare列少量缺失

**缺失值的处理：**

- 对于 `Age` 列：由于缺失值较多，因此不能直接删除缺失列，使用其他特征预测年龄。
- 对于 `Cabin` 列：由于缺失值较多，而且该特征较分散、随机。可以考虑将缺失和非缺失分为两个类别，或者在分析中不使用这一列。我选择分为缺失、非缺失两个类别。
- 对于 `Embarked` 列：由于缺失值较少，考虑使用众数填充。
- 对于 `Fare` 列：可以使用均值或中位数填充缺失值。

处理缺失值之后：

<img src="images/data_fixed.png" alt="data_fixed" style="zoom:50%;" />



#### 2. 数据编码

针对非数值型数据。

* 映射为唯一的整数。可以使用`sklearn.processing.LabelEncoder()`,每个不同的类别都被映射为一个唯一的整数值。但是，`LabelEncoder` 会在对具有大小关系的类别进行编码时引入错误的大小关系

* 映射为独热码。若某些非数值数据具有大小关系，如"small"、"big"，此时应该使用One-Hot Encoding。但是若使用 OneHotEncoder，会导致特征数量过多，可能影响模型训练速度



#### 3. 归一化(可选)

（针对连续型的数值型数据），对 Age 和 Fare 列进行归一化，因为这两列的数值范围较大。

可以使用`sklearn.processing.MinMaxScaler`或者`sklearn.preprocessing.StandardScaler()`



处理前后的数据：

<img src="images/data_fixed_view.png" alt="data_fixed_view" style="zoom:50%;" />



也可以使用sklearn自带的特征工程函数来完成以上**缺失值处理、编码、归一化等**操作。



#### 4. 划分数据集

可使用train_test_split函数

```python
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=<split_size>, random_state=0)
```



#### 5. 训练，拟合

使用sklearn的逻辑回归



#### 6. 输出预测结果，不计算指标，因无标准答案

![result](images/predict_result.png)

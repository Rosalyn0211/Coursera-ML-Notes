# 单变量线性回归(Linear Regression with One Variable)

### 模型表示
以房价预测问题为例，数据集包含不同房屋尺寸所售出的价格。    

我们将要用来描述这个回归问题的标记如下:

$m$ 代表训练集中实例的数量

$x$ 代表特征/输入变量

$y$ 代表目标变量/输出变量

$\left( x,y \right)$ 代表训练集中的实例

$({{x}^{(i)}},{{y}^{(i)}})$ 代表第$i$ 个观察实例

$h$ 代表学习算法的解决方案或函数也称为假设（hypothesis）

对于我们的房价预测问题，我们该如何表达 $h$？   

一种可能的表达方式为：$h_\theta \left( x \right)=\theta_{0} + \theta_{1}x$，因为只含有一个特征/输入变量，因此这样的问题叫作单变量线性回归问题。

<br>

### 代价函数
我们的目标便是选择出可以使得建模误差的平方和最小的模型参数$\theta_{0}$ 和 $\theta_{1}$。 
即代价函数$J \left( \theta_0, \theta_1 \right) = \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right)^{2}$最小。

<br>

### 梯度下降
梯度下降是一个用来求函数最小值的算法，我们将使用梯度下降算法来求出代价函数$J(\theta_{0}, \theta_{1})$ 的最小值。
梯度下降背后的思想是：开始时我们随机选择一个参数的组合$\left( {\theta_{0}},{\theta_{1}},......,{\theta_{n}} \right)$，计算代价函数，然后我们寻找下一个能让代价函数值下降最多的参数组合。
我们持续这么做直到找到一个局部最小值（local minimum），因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是全局最小值（global minimum），选择不同的初始参数组合，可能会找到不同的局部最小值。
批量梯度下降（batch gradient descent）算法的公式为：

其中$a$是学习率（learning rate），它决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大，
在批量梯度下降中，我们每一次都同时让所有的参数减去学习速率乘以代价函数的导数。

梯度下降中，我们要更新${\theta_{0}}$和${\theta_{1}}$ ，当 $j=0$ 和$j=1$时，会产生更新，所以你将更新$J\left( {\theta_{0}} \right)$和$J\left( {\theta_{1}} \right)$。
实现梯度下降算法的微妙之处是，在这个表达式中，如果你要更新这个等式，你需要同时更新${\theta_{0}}$和${\theta_{1}}$。

<br>

### 梯度下降的线性回归

对我们之前的线性回归问题运用梯度下降法，关键在于求出代价函数的导数，即：

$\frac{\partial }{\partial {{\theta }{j}}}J({{\theta }{0}},{{\theta }{1}})=\frac{\partial }{\partial {{\theta }{j}}}\frac{1}{2m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}^{2}}$

$j=0$ 时：$\frac{\partial }{\partial {{\theta }{0}}}J({{\theta }{0}},{{\theta }{1}})=\frac{1}{m}{{\sum\limits{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}}$

$j=1$ 时：$\frac{\partial }{\partial {{\theta }{1}}}J({{\theta }{0}},{{\theta }{1}})=\frac{1}{m}\sum\limits{i=1}^{m}{\left( \left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

则算法改写成：

Repeat {

 ${\theta_{0}}:={\theta_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{ \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}$

 ${\theta_{1}}:={\theta_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

 }
 
 ### python实现
 ##### Step 1 数据预处理
- 导入数据集
- 处理缺失数据
- 划分数据集

```
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : ,  : 1 ].values
Y = dataset.iloc[ : , 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/4,random_state=0)

```


##### Step 2 利用简单线性回归模型训练
我们使用sklearn.linear_model库中的LinearRegression类的fit()方法，将regressor对象对数据集进行训练。

```
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
```


#####  Step 3 预测结果
预测测试集的观察结果，将输出保存至Y_pred。
我们利用LinearRegression类的预测方法进行预测。

```
Y_pred = regressor.predict(X_test)
```

#####  Step 4 可视化
我们使用matplotlib.pyplot库对我们的训练集结果和测试集结果绘制散点图。

```
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()

plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()
```

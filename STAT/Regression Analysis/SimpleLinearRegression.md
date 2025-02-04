# Simple Linear Regression

## Introduction

### Model Formulation

我们可以用线性回归来衡量两个变量之间的相关关系. 形式上, 一个 simple linear regression model 可以写成:
$$
Y = \beta_0 + \beta_1X + \epsilon
$$
其中 
- $Y$ 是 dependent variable 或 response variable. 是我们关注的希望进行预测或者解释的变量. 在实践中是已知的. 
- $X$ 是 independent variable 或 predictor 或 explanatory variable 或 covariate. 是我们用来预测或者解释 dependent variable 的变量. 在实践中是已知的, 且至少目前我们认为是非随机的.
  - 这里的随机性与否我们可以如下解读: 假设我们在进行一个问卷调查, 关注 $Y$ (收入) 与 $X$ (工作年限) 之间的关系. 一种简单的处理方式是将 $X$ 视为非随机的, 对于每个个体都是一个确定的值. 但另一个角度, 按照实验设计的角度, 我们事实上在进行问卷调查时, 并不能知道下一个被调查者的工作年限是多少, 因此在这种意义上 $X$ 也是随机的. 不过为了问题的简化, 我们暂时将 $X$ 视为非随机的. 相应的 $X\beta$ 项也可以被视为一个确定的值. 换言之, 我们认为 $Y$ 的随机性是全部由 $\epsilon$ 引起的. 
- $\beta_0, \beta_1$ 是 model parameters. 其取值反映了 dependent variable 和 independent variable 之间的关系, 是我们希望通过数据估计的量.
  - $\beta_1$ 是斜率项, 表示了当 $X$ 变化一个单位时, $Y$ 的变化量.
  - $\beta_0$ 是截距项. 如果数据中的 $X$ 的范围是包括 0 的, 那么 $\beta_0$ 反映了当 $X=0$ 时, $Y$ 的值. 如果数据中的 $X$ 的范围不包括 0 (如 $X$ 是一个城市的人口, 那么 $X$ 一般意义上不会是 0), 那么 $\beta_0$ 不具有实际经济意义. 
- $\epsilon$ 是误差项. 是一个随机变量, 反映了我们的模型中一切未能解释的因素, 包括所有除 $X$ 以外对 $Y$ 有影响的因素，以及人类行为的随机性。
  
我们也将 $Y(X) = \beta_0 + \beta_1X$ 称为 regression line 或者 regression function. 因为这个式子在数学上相当于一条直线, 其反映了我们所给出的 dependent variable 和 independent variable 之间的关系的**整体估计**.

如果具体到每个数据点(观测值) $(Y_1, X_1), (Y_2, X_2), \ldots, (Y_n, X_n)$, 我们可以将 simple linear regression model 写成:
$$
Y_i = \beta_0 + \beta_1X_i + \epsilon_i, \quad i = 1, 2, \ldots, n
$$
其中 $\epsilon_i$ 是第 $i$ 个观测值的误差项.

> ### **OLS 与 Econometric**
> 
> 夹带私货, 稍微补充一点关于 Economics 的视角与思想, 因为线性回归本身也是计量经济学的基础. 内容主要引自陈强老师的计量经济学课程讲义. (侵删)
> 
> Econometric 是运用概率统计方法对经济变量之间的(因果)关系进行定量分析的科学。由于实验数据的缺乏，**计量经济学常常不足以确定经济变量之间的因果关系** (尽管我们真正在乎的是因果关系, 即 $X$ 是否导致了 $Y$). 
>
> - *例(相关关系)* 你看到街上的人们带雨伞，于是预测今天要下雨。这只是相关关系，“人们带伞”并不导致“下雨”。
> 
> 如果只对预测感兴趣，则相关关系就足够了。 如果要推断变量之间的因果关系，则计量分析必须建立在经济理论的基础之上，即在理论上存在 $X$ 导致 $Y$ 的作用机制。
>
> 因果关系难以确定的原因有很多:
> - 可能存在“逆向因果关系”(reverse causality)或“双向因果关系”
>   - *例(逆向因果)*: 收入增加引起消费增长, 而消费增长也会提高收入。
> - 可能存在“遗漏变量”(omitted variable)或“遗漏因素”(omitted factor)
>   - *例(遗漏变量)*: [一个相当有趣的例子!] 某外星人来到地球，发现人类会死亡，十分不解。于是开始在全球广泛观察死亡现象，并收集了大量的数据。结果发现，许多人类躺在医院病床( $X$ )之后死去( $Y$ )，故推断 医院病床是死亡的原因。外星人认为，由于躺在医院病床上，总是发生于死亡之前， 故不可能存在逆向因果关系。外星人于是将研究报告投稿发表于某顶尖经济学期刊，并在文末给出政策建议“珍爱生命，远离病床”。(这显然是荒诞的, 因为遗漏了一个重要的变量: 病人的病情)
>   - 这也是为什么我们要引入误差项(或称扰动项) $\epsilon$. 扰动项就像是一个“垃圾桶”，所有你不想要, 无法把握的东西都往里面扔. 但我们又希望扰动项拥有很好的性质 (或者说我们希望这个我们无法观测的项是在我们的掌控之中的). 尽管在很多情况下, 这听起来是自相矛盾的. 计量的很多玄妙之处就在于扰动项. 如果真正理解扰动项, 就加深了对计量经济学的理解。

### Regression Modelling Procedure

在进一步讨论回归分析的具体技术细节之前, 首先给出一个较为通用的回归分析的步骤. 首先我们要通过有关研究领域的相关理论知识, 以及对数据的初步观察, 来确定我们最初的回归模型设定. 然后我们可以使用一些数学方法来估计模型的参数 (粗略来讲可以包括通过推导进行分析得到的解析解, 以及通过数值模拟或类似梯度下降等方法进行的数值解). 接着是对模型的适用性进行评估, 例如参数是否统计学上显著, 模型的假设是否得到满足, 模型整体的拟合程度和解释性如何等等. 若模型的表现不佳, 则需要重新考虑模型的设定, 或者对数据进行进一步的处理等, 直到得到一个满意的模型.最终是模型验证, 通过对模型的预测能力进行验证, 来检验模型的有效性, 也包括从一系列可接受的模型中选择最优的模型等. 

这里的每个部分都有自己的相应方法和技术细节, 整体的流程也是围绕着这些方面展开的.

![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202501262235044.png)

### Model Assumptions (Gauss-Markov Assumptions)

为了计算和估计的方便, 我们常需要对 simple linear regression model 做一些假设. 这些假设通常被合称为 Gauss-Markov Assumptions. 这些假设包括:
1. $X_i$ 与 $\epsilon_i$ 独立. 
2. $\mathbb{E}(\epsilon_i) = 0$. 误差项的期望是 0. 
3. $\text{Cov}(\epsilon_i, \epsilon_j) = 0, \forall i \neq j$. 不同观测值的误差项之间两两互不相关. 也就是说这一个样本的误差项不会影响到另一个样本的误差项.
4. $\text{Var}(\epsilon_i) = \sigma^2, \forall i$. 误差项的方差是相同的 (homoscedasticity).
5. $\epsilon_i \sim \mathcal{N}(0, \sigma^2), \forall i$. 正态性假设. 

> 注: 在实践中, 正态性的假设并不是必须的. 其存在的意义主要是为了方便我们进行推断. **在后面的推导过程中, 需要应用到正态性假设的地方会特别指出. 故在其余地方可认为是不依赖于正态性假设的.**


## Model Estimation

在设定好一个 simple linear regression model 之后, 我们需要对模型的参数进行估计. 也就是我们需要估计 $\beta_0, \beta_1$ 以及 $\sigma^2$. 

### 系数 $\beta_0, \beta_1$ 的 Ordinary Least Squares Estimation 普通最小二乘估计 (OLS) 求解

一个最直接的想法就是如何估计 $\beta_0, \beta_1$.

***1. 构建最小二乘估计函数***

- 在 simple linear regression model 中, 我们的目标是估计 $\beta_0, \beta_1$. 为了估计这两个参数, 我们可以使用 OLS 方法. OLS 方法的目标是通过改变参数的估计值 $\hat{\beta}_0, \hat{\beta}_1$ , 使得我们的模型的 $Y$ 预测值 (记为 $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1x_i$) 与实际观测值 $y_i$ 之间的误差最小:
    $$
    \min_{\beta_0,\beta_1} \mathcal Q = \sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)^2
    $$

***2. 微积分求解最小二乘最小值***

- 这相当于一个自变量为 $\beta_0, \beta_1$ 的二元函数求极值的问题. 一个直接的方法是对 $\mathcal Q$ 分别关于 $\beta_0, \beta_1$ 求导, 令导数为 0, 解出 $\hat{\beta}_0, \hat{\beta}_1$ (严格意义上求解的是驻点, 但是由于 $\mathcal Q$ 是一个二次型, 因此这个驻点就是极值点).

- 由微积分的知识, 可以求导如下:
    $$  \begin{aligned} 
    \frac{\partial \mathcal Q}{\partial \beta_0} &= \frac{\partial}{\partial \beta_0} \sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)^2 = -2\sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i) = 0 \quad \spadesuit\\
    \frac{\partial \mathcal Q}{\partial \beta_1} &= \frac{\partial}{\partial \beta_1} \sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)^2 = -2\sum_{i=1}^n x_i(y_i - \beta_0 - \beta_1x_i) = 0 \quad \clubsuit
    \end{aligned}$$

- 有时也会将上面这两个方程组记为 Normal Equations. 对这个方程组化简, 可以得到:
    $$  \begin{aligned}
    \sum_{i=1}^n y_i &= n\beta_0 + \beta_1\sum_{i=1}^n x_i  \\
    \sum_{i=1}^n x_iy_i &= \beta_0\sum_{i=1}^n x_i + \beta_1\sum_{i=1}^n x_i^2
    \end{aligned}$$


- 这个方程组按照初等数学的视角完全可以当做是一个关于 $\beta_0, \beta_1$ 的二元一次方程组, 直接正常求解化简即可. 
- 为与后面的多元线性回归相呼应, 这里引入矩阵的表示进行求解. 
  - 事实上, 可以将上面的方程组写成矩阵的形式 (可以通过反推展开这个矩阵乘法来验证):
    $$
    \begin{bmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2 \end{bmatrix} \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix} = \begin{bmatrix} \sum y_i \\ \sum x_iy_i \end{bmatrix}
    $$
  - 因此我们可以直接将左边的矩阵求逆左乘右边的矩阵来求解 $\beta_0, \beta_1$ (若逆存在):
    $$
    \begin{bmatrix} \hat{\beta}_0 \\ \hat{\beta}_1 \end{bmatrix} = \begin{bmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2 \end{bmatrix}^{-1} \begin{bmatrix} \sum y_i \\ \sum x_iy_i \end{bmatrix}
    $$
  - 而这个 $2\times 2$ 的矩阵的逆可以直接计算出来 (回顾公式 $\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$):
    $$
    \begin{bmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2 \end{bmatrix}^{-1} = \frac{1}{n\sum x_i^2 - (\sum x_i)^2} \begin{bmatrix} \sum x_i^2 & -\sum x_i \\ -\sum x_i & n \end{bmatrix}
    $$
- 因此我们可以得到 $\hat{\beta}_0, \hat{\beta}_1$ 的表达式:
    $$\begin{aligned}
    \begin{bmatrix} \hat{\beta}_0 \\ \hat{\beta}_1 \end{bmatrix} &= \frac{1}{n\sum x_i^2 - (\sum x_i)^2} \begin{bmatrix} \sum x_i^2 & -\sum x_i \\ -\sum x_i & n \end{bmatrix} \begin{bmatrix} \sum y_i \\ \sum x_iy_i \end{bmatrix} \\
    &= \frac{1}{n\sum x_i^2 - (\sum x_i)^2} \begin{bmatrix} \sum x_i^2\sum y_i - \sum x_i\sum x_iy_i \\ n\sum x_iy_i - \sum x_i\sum y_i \end{bmatrix}
    \end{aligned}$$
- 也就是分别得到:
    $$
    \hat{\beta}_1 = \frac{n\sum x_iy_i - \sum x_i\sum y_i}{n\sum x_i^2 - (\sum x_i)^2}, \quad \hat{\beta}_0 = \frac{\sum x_i^2\sum y_i - \sum x_i\sum x_iy_i}{n\sum x_i^2 - (\sum x_i)^2}
    $$

- 我们的求解就完成了! 

***3. 系数估计结果的化简与整理***

- 不过这可能和我们常见的形式有些不同. 其实可以证明它们都是等价的. 下面直接给出一个更常见的形式:
    $$\begin{equation*}
    \boxed{\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}   = \frac{\text{Cov}(x, y)}{\text{Var}(x)}  = \hat\rho_{xy}\frac{\hat\sigma_y}{\hat\sigma_x}}
    \end{equation*}$$

    $$\begin{equation*}\boxed{
    \hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}}
    \end{equation*}$$

    - 这里的结果很直观. $\hat{\beta}_1$ 可以看作是 $x, y$ 的协方差除以 $x$ 的方差. 相当于衡量 $X$ 自己产生一单位的波动变化时, $Y$ 会跟随 $X$ 有多少的波动. $\hat{\beta}_0$ 则是在 $X$ 的均值处的 $Y$ 的均值.

    > 这里省略了具体的推导过程, 事实上上述过程只是代数上的单纯计算化简. 这里提供在验证演算过程中可能会计算用到的几个中间公式:
    > - $\bar x = \frac{1}{n}\sum_{i=1}^n x_i, \quad \bar y = \frac{1}{n}\sum_{i=1}^n y_i$
    > - $\sum_{i=1}^n (x_i - \bar{x})^2 = \sum_{i=1}^n x_i^2 + n\bar{x}^2 - 2\bar{x}\sum_{i=1}^n x_i = \sum_{i=1}^n x_i^2 - n\bar{x}^2 = \sum_{i=1}^n x_i^2 - n\left(\frac{1}{n}\sum_{i=1}^n x_i\right)^2 = \sum_{i=1}^n x_i^2 - \left(\sum_{i=1}^n x_i\right)^2/n$
    > - $\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y}) = \sum_{i=1}^n x_iy_i +n\bar{x}\bar{y} - \sum_{i=1}^n x_i\bar{y} - \sum_{i=1}^n y_i\bar{x} = \sum_{i=1}^n x_iy_i + n\bar{x}\bar{y} - \bar{y}\bar{x} n - \bar{x}\bar{y}n = \sum_{i=1}^n x_iy_i - n\bar{x}\bar{y}$

###  $\hat{\beta}_0, \hat{\beta}_1$ 的矩估计

- 回顾 simple linear regression model 的形式: 
    $$Y= \beta_0 + \beta_1X + \epsilon, $$
    - 我们对这个方程两边取期望, 可以得到:
        $$\begin{aligned}
        \mathbb{E}(Y) &= \mathbb{E}(\beta_0 + \beta_1X + \epsilon) = \beta_0 + \beta_1\mathbb{E}(X) + \mathbb{E}(\epsilon) = \beta_0 + \beta_1 X 
        \end{aligned}$$
        最后一个式子成立是因为 (a) 我们认为 $X$ 是非随机的, 因此期望等于本身 $\mathbb{E}(X) = X$;  (b) 误差项的期望是 0, i.e. $\mathbb{E}(\epsilon) = 0$.
        - 由于对 $X$ 的非随机性的假设, 我们得到了上面的期望表达式. 然而在许多地方, 我们也能看到诸如 $\mathbb{E}(Y|X) = \beta_0 + \beta_1X, \text{Var}(Y|X) = \sigma^2$ 这样的条件期望和条件方差的表达式. 这相当于在数学表达上通过给定 $X$ 这种条件下的期望和方差来表明其非随机性. 在某种意义上这种表达方法也是更为严谨的. 
    - 再通过把 $Y = \beta_0 +\beta_1 X +\epsilon$ 带入 $\text{Cov} (X,Y)$ 来计算协方差:
        $$\begin{aligned}
        \text{Cov}(X, Y) &= \text{Cov}(X, \beta_0 + \beta_1X + \epsilon) \\
        &= \text{Cov}(X, \beta_0) + \text{Cov}(X, \beta_1X) + \text{Cov}(X, \epsilon) \quad\scriptstyle{\text{(由Cov的线性性质)}} \\
        &= 0 + \beta_1\text{Cov}(X, X) + 0  = \beta_1\text{Var}(X)
        \end{aligned}$$
        其中倒数第二个等号中的两个 $0$ 是因为 (a) $\beta_0$ 和 $X$ 都是非随机的, 因此协方差为 0; (b) 由 Gauss-Markov 假设 1, $X$ 和 $\epsilon$ 独立, 因此协方差为 0.
    - 综上, 从 $\mathbb{E}(Y)$ 和 $\text{Cov}(X, Y)$ 的表达式中反解出 $\beta_0, \beta_1$ 的矩估计量 $\hat{\beta}_0, \hat{\beta}_1$:
        $$\begin{aligned}
        \hat{\beta}_1 &= \frac{\text{Cov}(X, Y)}{\text{Var}(X)}  \\
        \hat{\beta}_0 &= \bar{y} - \hat{\beta}_1\bar{x}
        \end{aligned}$$
        这和我们之前的 OLS 估计是一致的.

### OLS 的参数估计的一些有用结论

这里给出一些 OLS 估计的一些有用结论并给出(较为繁琐的数学证明). 这些结论主要是在后续的推到证明中会经常用到. 同样还是考虑 simple linear regression model $Y = \beta_0 + \beta_1X + \epsilon$.

- **残差之和恒为 0**, 即 $\sum_{i=1}^n \hat{\epsilon}_i = 0$. 
  - 这个结论是可以通过 OLS 的求解过程得到的. 回忆在上面的过程中, 我们希望回归模型的误差平方和 $\mathcal{Q} = \sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)^2$ 最小. 为了求解这个最小值, 我们对 $\mathcal{Q}$ 分别关于 $\beta_0, \beta_1$ 求导, 令导数为 0. 而在这过程中, 对于 $\beta_0$ 的导数为 0 的方程即为 (上文的 $\spadesuit$ 式子):
    $$\frac{\partial \mathcal Q}{\partial \beta_0} = -2\sum_{i=1}^n (y_i - \hat\beta_0 - \hat\beta_1x_i) = 0 \quad \spadesuit$$
    对其整理就有:
    $$\begin{aligned}
    \spadesuit \Rightarrow 
     \sum_{i=1}^n \left(y_i - \hat y_i\right) &= 0 \\
     \Rightarrow \sum_{i=1}^n \hat\epsilon_i &= 0
    \end{aligned}$$

- **观测值之和等于拟合值之和**, 即 $\sum_{i=1}^n y_i = \sum_{i=1}^n \hat{y}_i$.
  (上面的证明已经给出了这个结论)
- **OLS回归线总是通过 $(\bar{x}, \bar{y})$**, 即 $\hat{\beta}_0 + \hat{\beta}_1\bar{x} = \bar{y}$. (这可以通过对 $\spadesuit$ 的化简结果 $\sum_{i=1}^n y_i = n\beta_0 + \beta_1\sum_{i=1}^n x_i$两边同时除以 $n$ 得到)
- **每个观测的真实自变量 $x_i$ 乘以残差 $\hat{\epsilon}_i$ 的和为0**, 即 $\sum_{i=1}^n x_i\hat{\epsilon}_i = 0$. 
  - 这是由$\mathcal{Q}$ 对 $\beta_1$ 求导得到的 (上文的 $\clubsuit$ 式子):
    $$\frac{\partial \mathcal Q}{\partial \beta_1} = -2\sum_{i=1}^n x_i(y_i - \hat\beta_0 - \hat\beta_1x_i) = 0 \quad \clubsuit$$
    对其整理就有:
    $$\begin{aligned}
    \clubsuit \Rightarrow 
     \sum_{i=1}^n x_i(y_i - \hat y_i) &= 0 \\
     \Rightarrow \sum_{i=1}^n x_i\hat\epsilon_i &= 0
    \end{aligned}$$
- **每个样本的模型拟合值 $\hat y_i$ 乘以残差 $\hat{\epsilon}_i$ 的和为0**, 即 $\sum_{i=1}^n \hat{y}_i\hat{\epsilon}_i = 0$. 
  - 这是由上面已经证明出的 $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1x_i$ 以及 $\sum_{i=1}^n \hat{\epsilon}_i = 0$ , $\sum_{i=1}^n x_i\hat{\epsilon}_i = 0$ 可以得到的:
    $$\begin{aligned}
    \sum_{i=1}^n \hat{y}_i\hat{\epsilon}_i &= \sum_{i=1}^n (\hat{\beta}_0 + \hat{\beta}_1x_i)\hat{\epsilon}_i \\
    &= \sum_{i=1}^n \left(\hat{\beta}_0\hat{\epsilon}_i + \hat{\beta}_1x_i\hat{\epsilon}_i\right) \\
    &= \hat{\beta}_0\sum_{i=1}^n \hat{\epsilon}_i + \hat{\beta}_1\sum_{i=1}^n x_i\hat{\epsilon}_i \\
    &= 0
    \end{aligned}$$
  

### 误差方差 $\sigma^2$ 的估计

***点估计***

除了估计 $\beta_0, \beta_1$ 之外, 我们往往还比较关心误差项的波动大小. 回忆我们在 Gauss-Markov Assumptions 中假设了 $\epsilon_i$ 的方差是相同的, 且为 $\sigma^2$. (甚至在理论上我们还希望 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$). 这个小节我们就是希望给出 $\sigma^2$ 的估计.

> Note: 后面还将讨论估计量 $\hat{\beta}_0, \hat{\beta}_1$ 的方差. 但它们是不一样的. 这里的 $\sigma^2$ 是误差项的方差, 表示的是我们当前这个回归模型对于数据的解释的波动性大小. 而后面的 $\text{Var}(\hat{\beta}_0), \text{Var}(\hat{\beta}_1)$ 是估计量的方差, 表示的是我们用譬如OLS进行估计的时候对于参数估计的不确定性大小.

下给出具体推导:

- 按照矩估计的思路, 由于在理论上根据我们的假设, $\mathbb{E}(\epsilon_i) = 0, \text{Var}(\epsilon_i) = \sigma^2$, 因此 $\mathbb{E}(\epsilon_i^2) = \sigma^2$. 故我们可以用 $\epsilon_i^2$ 的样本均值 $\frac{1}{n}\sum_{i=1}^n {\epsilon}_i^2$ 来估计 $\sigma^2$.
  - 然而根据误差项的定义, 误差项我们是无法直接观测到的. 因此我们需要用残差 (residual) 来作为误差项的估计. 然后用 $\sum\hat\epsilon^2$ 来代替上面的 $\sum \epsilon_i^2$ 最终给出 $\sigma^2$ 的估计. 
- **残差是我们的模型预测值和实际观测值之间的差异, 数学上定义为:**
    $$\hat{\epsilon}_i = y_i - \hat{y}_i = y_i - \hat{\beta}_0 - \hat{\beta}_1x_i$$
    某种意义上, **残差是误差的实现(realization)**.
- 通过矩阵形式下较为繁琐的数学证明 (该证明可能会在后面单独补充), 可以求得:
    $$\mathbb{E}(\sum_{i=1}^n \hat{\epsilon}_i^2) = \mathbb{E}(\sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1x_i)^2) = (n-2)\sigma^2$$
    因此我们要将这个残差的平方和除以 $n-2$ 来得到 $\sigma^2$ 的无偏估计:
    $$\boxed{\hat{\sigma}^2 = \frac{1}{n-2}\sum_{i=1}^n \hat{\epsilon}_i^2}$$
    进一步用平方根来估计 $\sigma$ (其也往往被称为**回归的标准误差 (standard error of the regression) 或残差标准误(residual standard error)**):
    $$\boxed{\hat{\sigma} = \sqrt{\frac{1}{n-2}\sum_{i=1}^n \hat{\epsilon}_i^2}}$$
- 因此在缺少更多先验信息的情况下, 这就是我们对于误差项方差的估计. 不过注意这个估计是模型依赖的, 因此如果我们错误的假设了模型, 那么这个$\hat{\sigma}^2$ 也可能无法给出一个准确的估计.

***区间估计***

可以进一步证明, $\frac{(n-2)\hat{\sigma}^2}{\sigma^2} \sim \chi^2_{n-2}$, 因此可以计算概率:
$$\begin{aligned}
\mathbb{P} \left( \chi^2_{\alpha/2, n-2} \leq \frac{(n-2)\hat{\sigma}^2}{\sigma^2} \leq \chi^2_{1-\alpha/2, n-2} \right) &= 1 - \alpha \\
\end{aligned}$$

反过来就可以得到 $\sigma^2$ 的置信区间:
$$\begin{aligned}
\mathbb{P} \left( \frac{(n-2)\hat{\sigma}^2}{\chi^2_{1-\alpha/2, n-2}} \leq \sigma^2 \leq \frac{(n-2)\hat{\sigma}^2}{\chi^2_{\alpha/2, n-2}} \right) &= 1 - \alpha \\
\end{aligned}$$
即:
$$\begin{aligned}\boxed{
CI_{\sigma^2} = \left( \frac{(n-2)\hat{\sigma}^2}{\chi^2_{1-\alpha/2, n-2}}, \frac{(n-2)\hat{\sigma}^2}{\chi^2_{\alpha/2, n-2}} \right)}
\end{aligned}$$

其中 $\chi^2_{\alpha/2, n-2}$ 和 $\chi^2_{1-\alpha/2, n-2}$ 分别是自由度为 $n-2$ 的 $\chi^2$ 分布的 $\alpha/2$ 和 $1-\alpha/2$ 分位数. $\hat{\sigma}^2$ 的计算方法见上文.

### Simple Linear Regression 的极大似然估计

***在 MLE 的框架下, 我们需要对误差项的分布进行假设. 这里我们假设 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$, 即误差项是正态分布的.***

下给出极大似然估计的具体推导:
- 由于 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$, 且 $Y_i = \beta_0 + \beta_1X_i + \epsilon_i$, 因此 $Y_i \sim \mathcal{N}(\beta_0 + \beta_1X_i, \sigma^2)$  (这是因为这里的 $\beta_0 + \beta_1X_i$ 是一个确定的值, 因此就相当于给随机变量 $\epsilon_i$ 加上一个常数 $c$ 一样,  均值平移, 方差不变).
- 因此我们可以写出 $Y_i$ 的概率密度函数 (pdf):
    $$f(y_i) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \beta_0 - \beta_1x_i)^2}{2\sigma^2}\right)$$
- 趁机写出似然函数 (likelihood function):
    $$\begin{aligned}
    \mathcal L(\beta_0, \beta_1, \sigma^2) 
    &= \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y_i - \beta_0 - \beta_1x_i)^2}{2\sigma^2}\right) \\
    &= \left(\frac{1}{\sqrt{2\pi}\sigma}\right)^n \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)^2\right)
    \end{aligned}$$
- 以及对数似然函数 (最后一步省略常数项):
    $$\begin{aligned}
    \ell(\beta_0, \beta_1, \sigma^2) 
    &= \log \mathcal L(\beta_0, \beta_1, \sigma^2) \\
    &= -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)^2 \\
    &\propto -\frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)^2
    \end{aligned}$$
- 对于求解 $\beta_0, \beta_1$ 的极大似然估计, 我们可以直接对 $\ell$ 关于 $\beta_0, \beta_1$ 求导, 而这就是相当于对 $- \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)^2$ 求导, 这和 OLS 中的求解是完全一致的! 因此:
    $$
    \begin{aligned}
    \hat{\beta}_1^{{\text{MLE}}}= \hat{\beta}_1^{{\text{OLS}}}, ~
    \hat{\beta}_0^{{\text{MLE}}} = \hat{\beta}_0^{{\text{OLS}}}
    \end{aligned}
    $$
- 我们还可以直接利用这个对数似然求出方差的极大似然估计(把 $\sigma^2$ 当作一个整体来求导):
    $$\begin{aligned}
    \frac{\partial \ell}{\partial \sigma^2} &= -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (y_i - \beta_0 - \beta_1x_i)^2 = 0 \\
    \hat{\sigma}^2_{\text{ML}} &= \frac{1}{n}\sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1x_i)^2 = \frac{1}{n}\sum_{i=1}^n \hat{\epsilon}_i^2 = \frac{n-2}{n}\hat{\sigma}_{\text{OLS}}^2
    \end{aligned}$$
    不过也趁此看出, 极大似然估计的方差估计量是有偏的.



## Significance Testing, Confidence Intervals and Goodness of Fitting

在我们通过 OLS 或者 MLE 得到了 $\hat{\beta}_0, \hat{\beta}_1$ 以及 $\hat{\sigma}^2$ 之后, 我们往往会对这些估计量(尤其是 $\hat{\beta}_0, \hat{\beta}_1$) 进行检验. 对应的, 我们根据假设检验的过程也可以得到这些估计量的置信区间. 最终在模型整体的拟合程度上, 我们也会对模型的拟合程度进行评估. 这些方法都是对模型的适用性进行评估的重要手段.

### F-test for Model's Overall Significance (ANOVA)

首先我们可以对整个模型的显著性进行检验. 这个检验的目的是检验我们的模型总的而言是否能够对数据进行一个较好的解释. 而这个检验的方法就是 F-test, 也可以认为是方差分析 (ANOVA) 的一种形式. 尽管一般而言, F-test 在多元线性回归中更为常见, 但其基本思想在简单线性回归中也是适用的.


首先我们先直接给出下面三个表达式:
- **总平方和 (Total Sum of Squares, SST)**: 衡量了所有数据点相对于数据均值的波动程度 (其实就是样本方差不除以$n-1$的形式), SST 在给定数据后就可以直接计算出来, 与具体的模型无关. 其取值是每个数据点相对于数据均值之差的平方和:
    $$\text{SST} = \sum_{i=1}^n (y_i - \bar{y})^2$$
- **回归平方和 (Regression Sum of Squares, SSR)**: 衡量了回归模型对数据的解释程度, 也就是回归模型预测值相对于数据均值的波动程度. 其取值是每个数据点模型给出的拟合值相对于数据均值之差的平方和:
    $$\text{SSR} = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2$$
- **残差平方和 (Residual Sum of Squares, SSE)**: 衡量了模型无法解释的部分, 也就是残差的波动程度, 即我们刚刚用来估计 $\sigma^2$ 的部分, 其取值是每个数据点真实值相对于模型预测值之差的平方和:
    $$\text{SSE} = \sum_{i=1}^n \hat{\epsilon}_i^2 = \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

> ***Note:*** 这三个变量的名字在不同的地方称呼较为混乱, 因此要注重其具体含义和计算方式.并且要自己内部保持一致.

- 这三个变量之间有一个重要的恒等关系:
    $$\boxed{\text{SST} \equiv \text{SSR} + \text{SSE}}$$
    即
    $$\sum_{i=1}^n (y_i - \bar{y})^2 \equiv \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 + \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

  - 直观上这是非常自然的. 即数据的总的波动程度等于模型能够解释的波动的部分加上模型无法解释的波动的部分. 在统计学上, 某种意义上波动的部分也可以看作是信息的部分, 因此这这个恒等式也可以理解为模型能够解释的信息量加上模型无法解释的信息量等于总的信息量. 这显然是永远成立的.
  - 下面数学地证明这个恒等关系. 其过程较为琐碎, 因此也可以直接承认该恒等式的成立:
    - 首先我们恒有:
      $$ y_i - \bar{y} \equiv (\hat{y}_i - \bar{y}) + (y_i - \hat{y}_i)$$
    - 对上式两边平方并关于所有数据点求和, 可以得到:
      $$\begin{aligned}
      \sum_{i=1}^n (y_i - \bar{y})^2 &\equiv \sum_{i=1}^n \left[(\hat{y}_i - \bar{y})^2 + 2(\hat{y}_i - \bar{y})(y_i - \hat{y}_i) + (y_i - \hat{y}_i)^2 \right] \\
      \sum_{i=1}^n (y_i - \bar{y})^2 &\equiv \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 + 2\sum_{i=1}^n (\hat{y}_i - \bar{y})(y_i - \hat{y}_i) + \sum_{i=1}^n (y_i - \hat{y}_i)^2 \\
      \text{SST} &\equiv \text{SSR} + 2\sum_{i=1}^n (\hat{y}_i - \bar{y})(y_i - \hat{y}_i) + \text{SSE}
      \end{aligned}$$
    - 遍历所有的数据点, 对左右两侧进行求和, 并且参考刚才给出的各项定义, 可以得到:
      $$\begin{aligned}
      \sum_{i=1}^n (y_i - \bar{y})^2 &\equiv \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 + 2\sum_{i=1}^n (\hat{y}_i - \bar{y})(y_i - \hat{y}_i) + \sum_{i=1}^n (y_i - \hat{y}_i)^2 \\
      \text{SST} &\equiv \text{SSR} + 2\sum_{i=1}^n (\hat{y}_i - \bar{y})(y_i - \hat{y}_i) + \text{SSE}
      \end{aligned}$$
    - 下面的任务便是证明 $2\sum_{i=1}^n (\hat{y}_i - \bar{y})(y_i - \hat{y}_i) = 0$. 
      - 在该项的证明过程中, 需要用到前面已经证明过的两个结论: (1) $\sum_{i=1}^n \hat{\epsilon}_i = 0$; (2) $\sum_{i=1}^n \hat y_i\hat{\epsilon}_i = 0$.
      - 利用残差和为零, 我们可以很快的证明 $2\sum_{i=1}^n (\hat{y}_i - \bar{y})(y_i - \hat{y}_i) = 0$ :
          $$\begin{aligned}
          2\sum_{i=1}^n (\hat{y}_i - \bar{y}) {(y_i - \hat{y}_i)} &= 2\sum_{i=1}^n (\hat{y}_i - \bar{y}) {\hat \epsilon_i} \\
          &= 2\sum_{i=1}^n ( {\hat{y}_i \hat \epsilon_i} - \bar{y} {\hat \epsilon_i}) \\
          &= 2  {\sum_{i=1}^n \hat{y}_i \hat \epsilon_i} - 2\bar{y}  {\sum_{i=1}^n\hat \epsilon_i} \\
          &= 0
          \end{aligned}$$

- 利用这个恒等关系 $\text{SST} \equiv \text{SSR} + \text{SSE}$, 我们可以利用 ANOVA 的思想定义 F-test 统计量:
    $$\boxed{F_0 = \frac{\text{SSR}/\text{df}_{\text{SSR}}}{\text{SSE}/\text{df}_{\text{SSE}}} = \frac{\text{SSR}/1}{\text{SSE}/(n-2)} \triangleq \frac{\text{MSR}}{\text{MSE}} \sim \mathcal F(1, n-2)}$$
  - 直观上相当于在比较一个回归模型的**能够被模型解释的变异性/信息(SSR)** 和**模型无法解释的变异性/信息(SSE)** 的大小占比来判断模型的拟合程度. 也就是说, 如果模型能够解释的比例较大, 那么模型的拟合程度就较好. 这就是 F-test 的基本思想.
  - 不难发现 $F_0$ 的分母 $\text{MSE} = \text{SSE}/(n-2)$ 就是我们之前定义的 $\hat{\sigma}^2$. 这里将 SSR 和 SSE 分别除以自由度的结果定义为 $\text{MSR}$ 和 $\text{MSE}$, 分别称为**回归均方 (Mean Square Regression)** 和**残差均方 (Mean Square Error)**.
  - 数学上, 这个 F-test 作为假设检验的 Null Hypothesis 是: $H_0: \beta_1 = 0$, 而斜率系数为 0 即表示选取的变量对于因变量的解释能力为 0; 对应的 $H_1: \beta_1 \neq 0$, 即模型是能够解释数据的. 
    - 对于自由度的简要说明: $\text{df}_{\text{SSR}} = 1$, 这是因为其表达式可以整理为: $\text{SSR} = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 = \hat{\beta}_1 \sum_{i=1}^n y_i (x_i - \bar{x})$,  完全由 $\hat{\beta}_1$ 一个参数决定, 因此自由度为 1. $\text{df}_{\text{SSE}} = n-2$, 这是由于估计$\beta_0, \beta_1$ 两个参数时对残差$y_i - \hat{y}_i$ 有两个约束条件, 因此自由度为 $n-2$.而总的$\text{df}_{\text{SST}} = n-1$. 该项与样本的方差是一致的, 其自由度为 $n-1$. 事实上, 自由度也有同样的对应恒等关系, 即: $\text{df}_{\text{SST}} = \text{df}_{\text{SSR}} + \text{df}_{\text{SSE}}$.
  - 实践中, 我们在拟合好模型后即可计算 $F_0$ 的值, 然后根据给定的显著性水平 $\alpha$ 确定拒绝域的 critial value 为 $F_{\alpha}(1, n-2)$. 故若 $F_0 > F_{\alpha}(1, n-2)$, 则拒绝原假设, 即模型是显著的. 或者等价的考虑 $p$-value, $p = \mathbb{P}(F_0 > F_{\alpha}(1, n-2))$, 若 $p < \alpha$, 则拒绝原假设, 即模型是显著的.
  
### t-test for Individual Coefficients

***关于 $\beta_1$***

在 F-test 之后, 我们往往还会对模型中的参数进行检验. 这个检验的目的是检验我们的模型中的参数是否显著(显著地不等于0). 这个检验的方法就是 t-test. 对于 simple linear regression model, 我们主要关心的是斜率系数 $\beta_1$ 的显著性检验.


斜率系数$\beta_1$的显著与否 (在统计上的 $H_0: \beta_1 = 0$ vs $H_1: \beta_1 \neq 0$) 可以作如下理解:
- 如果不能拒绝 $H_0: \beta_1 = 0$ (即系数不显著), 则意味着 **$x$ 和 $y$ 之间没有线性关系**. 这既可能表示$x$对$y$基本没有任何解释能力(对于$y$的最优预测就是用自己的均值$\bar{y}$而与$x$无关), 也可能表示$x$和$y$之间的关系是非线性的.
  ![](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/202502011242875.png)
-  如果拒绝了 $H_0: \beta_1 = 0$ (即系数显著), 则意味着 **$x$ 和 $y$ 之间存在线性关系**. 这就意味着我们的模型是显著的, 且$x$对$y$有解释能力. 这也是我们希望看到的结果. 不过这只能说明线性的关系是 adequate的, 但不能说明是最优的, 因此可能一些非线性的关系依然是存在的.
  ![20250203192227](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250203192227.png) 

因此可以构建如下的 t-test 统计量对斜率系数进行检验:
$$\boxed{t_0 = \frac{\hat{\beta}_1}{\text{SE}(\hat{\beta}_1)}  \sim t(n-2)}$$
- 其中 $\text{SE}(\hat{\beta}_1)$ 是 $\hat{\beta}_1$ 的标准误, 可以被证明其具体计算公式为:
    $$\boxed{\text{SE}(\hat{\beta}_1) = \sqrt{\text{Var}(\hat{\beta}_1)} = \sqrt{\frac{\hat{\sigma}^2}{\sum_{i=1}^n (x_i - \bar x)^2}} = \sqrt{\frac{\sum_{i=1}^n(\hat y_i - y_i)^2 / (n-2)}{\sum_{i=1}^n (x_i - \bar x)^2}}}$$
- 当 $|t_0| > t_{\alpha/2}(n-2)$ 时, 我们拒绝原假设, 即认为 $\beta_1$ 是显著的. 或者等价的考虑 $p$-value, $p = \mathbb{P}(|t_0| > t_{\alpha/2}(n-2))$, 若 $p < \alpha$, 则拒绝原假设, 即认为 $\beta_1$ 是显著的.

需要指出的, 对于系数的 t-test 同样存在单侧的类型 ($H_0: \beta_1 \leq 0$ vs $H_1: \beta_1 > 0$ 或者 $H_0: \beta_1 \geq 0$ vs $H_1: \beta_1 < 0$), 这种检验的目的是检验系数的方向. 但在实践中, 一般都是双侧检验.

借着 t-test 的公式, 还可以对应写出其对应的置信区间:
$$
\boxed{\text{CI}_{\beta_1} = \hat{\beta}_1 \pm t_{\alpha/2}(n-2) \times \text{SE}(\hat{\beta}_1)}
$$

***关于 $\beta_0$***

我们可以同样对截距项 $\beta_0$ 进行 t-test. 其检验的目的是检验截距项是否显著. 其检验的 Null Hypothesis 是 $H_0: \beta_0 = 0$ vs $H_1: \beta_0 \neq 0$. 其检验的统计量为:
$${t_0 = \frac{\hat{\beta}_0}{\text{SE}(\hat{\beta}_0)}  \sim t(n-2)}$$
其中
$$\text{SE}(\hat{\beta}_0) = \sqrt{\text{Var}(\hat{\beta}_0)} = \sqrt{ \sigma^2 \left(\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right)}$$

然而在实践中, 其显著性并不是我们关心的重点. 


> ***NOTE !***
> 
> 事实上, 确定 $\beta_1 = 0$与否是一个非常重要的结论, 需要谨慎对待. 不论是 t-test 还是 F-test 都是一种辅助的工具. 
> 
> - 即使这两个检验都在暗示其不显著, 但是也不一定意味着 $y$ 和 $x$ 之间没有关系. 在统计学上不显著还有可能是因为 1. 数据量不够大; 2. 数据的噪声, 测量的方差较大; 3. 不合适的 $x$ 的取值范围掩盖了真实的关系; 4. 数据的非线性关系等等. 因此在实践中, 我们需要综合考虑多方面的因素来判断模型的适用性.
>
> - 反过来, 即使这两个检验都在暗示其显著, 但是其对应的系数取值却相对于数据的数量级非常小. 这时尽管其统计学意义上是显著的, 但是其实际的$x$的变化对于$y$的影响可能是微乎其微的. 这时常认为该变量在经济意义上不显著. 

### Goodness of Fitting / Coefficient of Determination $R^2$  

在我们对模型的参数进行检验之后, 我们往往还会对模型的拟合程度进行评估. 这个评估的指标就是 $R^2$ (Coefficient of Determination, 判决系数). $R^2$ 是一个介于 0 和 1 之间的数值, 其表示的是模型能够解释的数据的比例. 其定义为:
$$\boxed{R^2 = \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2} \in [0,1]} 
$$

- 回顾恒等式 $\text{SST} \equiv \text{SSR} + \text{SSE}$, 我们可以看出 $R^2$ 的定义是非常自然的. 其实就是模型能够解释的变异性的部分(可以理解为信息)占总的变异性部分的比例. 
- $R^2$ 越接近 1, 表示模型能够解释的数据越多, 拟合程度越好. 反之, $R^2$ 越接近 0, 表示模型的解释能力越差.
- 在 simple linear regression 中, $R^2 = \text{Corr}(x, y)^2 = \text{Corr}(\hat{y}, y)^2$, 即 $R^2$ 是自变量和因变量之间的相关系数的平方. (这个结论在多元线性回归中不再成立)

然而, 我们在使用 $R^2$ 时同样需要非常谨慎:

1. 我们总可以通过引入更多的变量来提高 $R^2$ 的值 (引入变量至少不会让 $R^2$ 变小, 通常会让 $R^2$ 变大). 然而引入过多的没有必要的变量并不是我们所希望的. 因此在多元的场合下, 我们还会引入 Adjusted $R^2$ 来对 $R^2$ 进行修正. 其具体定义见下一节.
2. $R^2$ 本身还和 $X$ 的分布有关. 一般而言, $R^2$ 在 $X$ 较为分散时会较大, 而在 $X$ 较为集中时会较小. 因此 $R^2$ 较大有时可能也只是由于 $X$ 的取值范围在一个过大的范围内所导致. 
3. 一般情况下, $R^2$ 不会是回归直线斜率的度量. 即使 $R^2$ 较大, 也不代表斜率是陡峭的.
4. $R^2$ 也不是模型适用性程度的度量, 即使 $Y$ 与 $X$ 是非线性相关的, 往往 $R^2$ 也会很大. 因此 $R^2$ 不能代表模型的适用性.

因此, 即使 $R^2$ 较大, 也并不必然意味着回归模型能够进行精确的数据预测 !


## Making Predictions

当我们经过以上的步骤, 得到了模型的参数估计以及模型的拟合程度确定这是一个可用的模型之后, 我们就可以利用这个模型来进行预测. 

假设我们有一个新的观测 $X^*$, 我们可以利用模型的预测值 $\hat{Y}^*$ 来对其进行预测. 其预测值的计算公式为:
$$\boxed{\hat{Y}^* = \hat{\beta}_0 + \hat{\beta}_1 X^*}$$

可以进一步给出其方差的计算公式:
$$\begin{aligned}
\text{Var}(\hat{Y}^*) &= \text{Var}(\hat{\beta}_0 + \hat{\beta}_1 X^*) \\
&= \sqrt{ \hat\sigma^2  (1 + \frac{1}{n} + \frac{(X^* - \bar{X})^2}{\sum_{i=1}^n (X_i - \bar{X})^2}) }
\end{aligned}$$

因此, 我们可以给出新的预测值的置信区间:
$$\boxed{\text{CI}_{\hat{Y}^*} = \hat{Y}^* \pm t_{\alpha/2}(n-2) \times \sqrt{ \hat\sigma^2  (1 + \frac{1}{n} + \frac{(X^* - \bar{X})^2}{\sum_{i=1}^n (X_i - \bar{X})^2}) }}$$

**NOTE: 这边还需要区别另一个置信区间, 即已有的数据$X$在回归中估计的$\hat Y$也有对应的置信区间. 直接给出其公式为:**
$$\boxed{\text{CI}_{\hat{Y}} = \hat{Y} \pm t_{\alpha/2}(n-2) \times \sqrt{ \hat\sigma^2  \left(\frac{1}{n} + \frac{(X - \bar{X})^2}{\sum_{i=1}^n (X_i - \bar{X})^2}\right) }}$$

## Statistical Discussion

上面的部分基本上已经给出了在实践中的 simple linear regression 的一些基本步骤.  下面这里我们对 simple linear regression 的一些统计学性质进行理论上的讨论. 

### OLS 中 $\hat\beta_0^{\tiny LS}, \hat\beta_1^{\tiny LS}$ 的无偏性

(这里用上标$^ {\tiny LS}$表示是由OLS估计得到的参数)

***Claim***: OLS 估计得到的 $\hat\beta_0^{\tiny LS}, \hat\beta_1^{\tiny LS}$ 是无偏估计, 即:  $\mathbb{E}(\hat\beta_0^{\tiny LS}) = \beta_0, \mathbb{E}(\hat\beta_1^{\tiny LS}) = \beta_1$.

***Proof***:

**对于 $\hat\beta_1^{\tiny LS}$:**
$$\begin{aligned}
\hat\beta_1^{\tiny LS} &= \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} \\
&= \frac{\sum_{i=1}^n (x_i - \bar{x})y_i}{\sum_{i=1}^n (x_i - \bar{x})^2} - \frac{\bar{y}\sum_{i=1}^n (x_i - \bar{x})}{\sum_{i=1}^n (x_i - \bar{x})^2} \\ &= \frac{\sum_{i=1}^n (x_i - \bar{x})y_i}{\sum_{i=1}^n (x_i - \bar{x})^2}  \quad \text{  (因为 $\sum_{i=1}^n (x_i - \bar{x}) = \sum_{i=1}^n x_i - n\bar{x} = 0$)}\\
&= \sum_{i=1}^n \frac{(x_i - \bar{x})}{\sum_{i=1}^n (x_i - \bar{x})^2} y_i \\ &\triangleq \sum_{i=1}^n c_i y_i \quad \text{  (其中 $c_i = \frac{(x_i - \bar{x})}{\sum_{i=1}^n (x_i - \bar{x})^2}$)}
\end{aligned}$$

即 $\hat\beta_1^{\tiny LS}$ 可以看作是 $y_i$ 的线性组合. 故求期望时:
$$\begin{aligned}
\mathbb{E}(\hat\beta_1^{\tiny LS}) &= \mathbb{E}\left(\sum_{i=1}^n c_i y_i\right) \\
&= \sum_{i=1}^n c_i \mathbb{E}(y_i) \\
&= \sum_{i=1}^n c_i (\mathbb{E}\beta_0 + \mathbb{E}\beta_1 x_i + \mathbb{E}\epsilon_i) \\
&= \sum_{i=1}^n c_i (\beta_0 + \beta_1 x_i) \\
&= \beta_0 \sum_{i=1}^n c_i + \beta_1 \sum_{i=1}^n c_i x_i \\
\end{aligned}$$

而其中
$$\begin{aligned}
\sum_{i=1}^n c_i &= \sum_{i=1}^n \frac{(x_i - \bar{x})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{1}{\sum_{i=1}^n (x_i - \bar{x})^2} \sum_{i=1}^n (x_i - \bar{x}) = 0 \\
\sum_{i=1}^n c_i x_i &= \sum_{i=1}^n \frac{(x_i - \bar{x})}{\sum_{i=1}^n (x_i - \bar{x})^2} x_i = \frac{\sum_{i=1}^n (x_i - \bar{x})x_i}{\sum_{i=1}^n (x_i - \bar{x})^2}  = \frac{\sum_{i=1}^n (x_i-\bar x)(x_i - \bar{x})}{\sum_{i=1}^n (x_i - \bar{x})^2} = 1
\end{aligned}$$

故 $\mathbb{E}(\hat\beta_1^{\tiny LS}) = \beta_1$.

**对于 $\hat\beta_0^{\tiny LS}$:**

由 $y_i  = \beta_0 + \beta_1 x_i + \epsilon_i$ 可以得到:
$$\begin{aligned}
\sum_{i=1}^n y_i &= \sum_{i=1}^n( \beta_0 + \beta_1 x_i + \epsilon_i) \\
\Rightarrow \bar{y} &= \beta_0 + \beta_1 \sum_{i=1}^n x_i / n + \sum_{i=1}^n \epsilon_i \ / n\ \\
\end{aligned}$$
因此
$$\begin{aligned}
 \mathbb{E}(\hat\beta_0^{\tiny LS}) &= \mathbb{E}(\bar{y} - \hat\beta_1^{\tiny LS} \bar{x} ) \quad \text{  (根据 $\hat\beta_0$ 的OLS结果)} \\
&= \mathbb{E} ( \beta_0 + \beta_1 \bar{x} + \frac{1}{n} \sum_{i=1}^n \epsilon_i - \hat\beta_1^{\tiny LS} \bar{x}) \quad \text{  (代入上面推导的 $\hat\beta_1^{\tiny LS}$)} \\
&= \beta_0 + \beta_1 \bar{x} - \bar{x} \mathbb{E}(\hat\beta_1^{\tiny LS}) + \frac{1}{n} \sum_{i=1}^n \mathbb{E}(\epsilon_i) \\
&= \beta_0 + \beta_1 \bar{x} - \bar{x} \beta_1 + 0 \quad \text{  (因为 $\mathbb{E}(\epsilon_i) = 0$, 且 $\mathbb{E}(\hat\beta_1^{\tiny LS}) = \beta_1$)} \\
&= \beta_0
\end{aligned}$$

### OLS 中 $\hat\beta_0^{\tiny LS}, \hat\beta_1^{\tiny LS}$ 的方差

前文已经给出了 $\hat\beta_0^{\tiny LS}, \hat\beta_1^{\tiny LS}$ 的方差的计算公式. 这里给出公式的推导过程:

***Claim***: OLS 估计得到的 $\hat\beta_0^{\tiny LS}, \hat\beta_1^{\tiny LS}$ 的方差为:
$$\begin{aligned}
\text{Var}(\hat\beta_0^{\tiny LS}) &= \sigma^2 \left(\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right) \\
\text{Var}(\hat\beta_1^{\tiny LS}) &= \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
\end{aligned}$$

***Proof***:

回顾: 我们前面已经证明 $\text{Var}(y_i) = \text{Var}(\beta_0 + \beta_1 x_i + \epsilon_i) = \text{Var}(\epsilon_i) = \sigma^2$.

**对于 $\hat\beta_1^{\tiny LS}$:**

我们已经知道, $\hat\beta_1^{\tiny LS}$ 可以写成 $y_i$ 的线性组合的形式:
$$\hat\beta_1^{\tiny LS} = \sum_{i=1}^n c_i y_i$$
其中 $c_i = \frac{(x_i - \bar{x})}{\sum_{i=1}^n (x_i - \bar{x})^2}$ (而这一部分我们认为是不具有随机性的, 因此在下面求方差的时候可以提出来)

因此, $\hat\beta_1^{\tiny LS}$ 的方差可以写成:
$$\begin{aligned}
\text{Var}(\hat\beta_1^{\tiny LS}) &= \text{Var}\left(\sum_{i=1}^n c_i y_i\right) \\
&= \sum_{i=1}^n c_i^2 \text{Var}(y_i) \quad \text{  (因为 $y_i$ 是独立的, 因此求和可以提到方差的外面)} \\
&= \sum_{i=1}^n c_i^2 \sigma^2 \\
&= \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
\end{aligned}$$

**对于 $\hat\beta_0^{\tiny LS}$:**

又由于 $\hat\beta_0^{\tiny LS} = \bar{y} - \hat\beta_1^{\tiny LS} \bar{x}$, 因此:
$$\begin{aligned}
\text{Var}(\hat\beta_0^{\tiny LS}) &= \text{Var}(\bar{y} - \hat\beta_1^{\tiny LS} \bar{x}) \\
&= \text{Var}(\bar{y}) + \bar{x}^2 \text{Var}(\hat\beta_1^{\tiny LS}) - 2\bar{x} \text{Cov}(\bar{y}, \hat\beta_1^{\tiny LS}) \\
&= \text{Var}(\bar{y}) + \bar{x}^2 \text{Var}(\hat\beta_1^{\tiny LS}) \quad \text{  (因为 $\bar{y}$ 和 $\hat\beta_1^{\tiny LS}$ 是独立的)} \\
&= \frac{\sigma^2}{n} + \bar{x}^2 \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2} \\
&= \sigma^2 \left(\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right)
\end{aligned}$$

### OLS 中 $\hat\beta_0^{\tiny LS}, \hat\beta_1^{\tiny LS}$ 的分布

如果我们对 $\epsilon_i$ 作出正态性假设: $\epsilon_i \stackrel{iid}{\sim} \mathcal N(0, \sigma^2)$, 那么我们可以得到 $\hat\beta_0^{\tiny LS}, \hat\beta_1^{\tiny LS}$ 的分布:
$$
\begin{aligned}
\hat\beta_0^{\tiny LS} &\sim \mathcal N(\beta_0, \sigma^2 \left(\frac{1}{n} + \frac{\bar{x}^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right)) \\
\hat\beta_1^{\tiny LS} &\sim \mathcal N(\beta_1, \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2})
\end{aligned}
$$

### OLS 中 $\hat\beta_0^{\tiny LS}, \hat\beta_1^{\tiny LS}$ 的 Gauss-Markov Theorem

***Theorem***:  对于满足假设 $\mathbb{E}(\epsilon_i) = 0, \text{Var}(\epsilon_i) = \sigma^2, \text{Cov}(\epsilon_i, \epsilon_j) = 0$ 的线性回归模型 $y_i = \beta_0 + \beta_1 x_i + \epsilon_i$, OLS 得到的 $\hat\beta_0^{\tiny LS}, \hat\beta_1^{\tiny LS}$ 是无偏的. 且与其他的对于 $\beta_0, \beta_1$ 的无偏估计相比, OLS 的估计具有最小的方差. 通常称OLS估计是**最佳线性无偏估计(Best Linear Unbiased Estimator, BLUE)**.

### OLS 中 $\sigma^2$ 的无偏性

***Claim***: OLS 估计得到的 $\sigma^2$ 是无偏估计, 即:  $\mathbb{E}(\hat\sigma^2) = \sigma^2$.

## Other Discussions

这里进行一些补充性讨论. 

### Regression with R

在 R 中, 我们可以使用 `lm()` 函数来进行线性回归的拟合. 该函数的基本使用方法为:
```R
X = c(1.0, 1.5, 2.1, 2.8, 3.4, 4.0) # 模拟一些自变量
Y = c(2.1, 3.2, 4.0, 5.0, 6.0, 7.0) # 模拟一些因变量
fit = lm(Y ~ X) # 进行线性回归的拟合
summary(fit) # 查看拟合的结果
```

最后 `summary(fit)` 的结果会给出拟合的结果, 包括了回归系数的估计, 拟合的 $R^2$ 等等:

![20250204174139](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250204174139.png)


# 7. Regularization and Model Selection

## 7.0 Regularization 
*[UCAS version](https://www.bilibili.com/video/BV1ga4y157L5?p=8&vd_source=8a00dab0be94d29388f2286892ba8d50)*

### Ridge Regression / Tikhanov (L2) Regularization

沿用Note4的记号，我们的目标是最小化：
$$\begin{aligned}
\min_{f_X} \quad & \frac1n \sum_{k=1}^n L(f_X(x_k), y_k) \\
\text{s.t.} \quad & f_X(x) \in \mathcal{F}
\end{aligned}$$

进过一定的泛函等技巧，并确定MSE为Loss的具体形式，以线性情况为例，可以得到:

$$\begin{aligned}
\min_{\theta} \quad & \frac1n \sum_{k=1}^n (y_k - \theta^T x_k)^2 
\\ \text{s.t.} \quad & \theta \in \Omega
\end{aligned}$$

而参考bias-variance tradeoff，我们需要尽可能简化模型的形式，而在经过泛函对应后，即为限制$\theta$的大小. 这里最开始是以二范数为例，即得到了 Tikhanov Regularization 的形式：
$$\begin{aligned}
\min_{\theta} \quad & \frac1n \sum_{k=1}^n (y_k - \theta^T x_k)^2 \\
\text{s.t.} \quad & ||\theta||_2^2 \leq r
\end{aligned}$$


对其具体形式进行进一步求解：

$$\begin{aligned}
\min_{\theta} \quad & (X\theta - y)^T(X\theta - y) \\
\text{s.t.} \quad & \theta^T\theta \leq r
\end{aligned}$$

再次引入Lagrange乘子，得到：
$$\begin{aligned}
L(\theta, \lambda) &= (X\theta - y)^T(X\theta - y) + \lambda(\theta^T\theta - r) \\ 
\nabla_{\theta} L(\theta, \lambda) &=  \cdots = 2X^TX\theta - 2X^Ty + 2\lambda\theta = 0 \\
\Rightarrow \quad & \theta_{\text{Ridge}} = (X^TX + \underbrace{\lambda I}_{\text{Diagnal Loading}})^{-1}X^Ty
\end{aligned}$$

**说明：**

- OLS的最小二乘解是Unbiased的，然而这里的结果是有偏的
- 正如bias-variance tradeoff所说，这里我们通过牺牲一定的bias，来降低variance，从而提高整体的泛化能力（因为variance是更无法控制的）


### SVD 与 (L2) Regularization

**SVD 原理**
- 对于对称矩阵    ($A=A^T$)
  $$ A = U\Lambda U^T = \sum_{i=1}^n \lambda_i u_i u_i^T$$
  其中$\Lambda=\text{diag}(\lambda_1,...,\lambda_n), UU^T=I$
  
- 对于Normal 矩阵  ($A^TA=AA^T$)
  $$ A = U^{-1}\Lambda U $$
  其中$\Lambda=\text{diag}(\lambda_1,...,\lambda_n), UU^{-1}=I$

- 对于一般方阵
  $$ A = U^{-1}\Lambda U $$
  其中$\Lambda=\text{diag}(J_1,...,J_k), J_k= \begin{bmatrix} \lambda_{k_1} & 0 & 0 & \cdots & 0 \\ 1 & \lambda_{k_2} & 0 & \cdots & 0 \\ 0 & 1 & \lambda_{k_3} & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & \lambda_{k_l} \end{bmatrix} $；这里称$\Lambda$为*Jordan Canonical Form*, $J_k$为*Jordan Block*

- **Singular Value Decomposition** - 对于一般矩阵 ($A \in \mathbb R^{m\times n}$) 
  $$ A = U\Sigma V^T $$
  其中$ UU^T = I, VV^T = I, \Sigma = \begin{bmatrix} \Lambda & 0 \\ 0 & 0 \end{bmatrix} $；且$\Lambda$为一个diag matrix，其阶对应着$A$的rank


**SVD 解释 L2 Regularization**

已知Ridge Regression 结果：
$$ \theta_{\text{Ridge}} = (X^TX + \lambda I)^{-1}X^Ty $$

对X (假设Column Full Rank)进行SVD：
$$\begin{aligned}
X &= U\Sigma V^T, \\ \text{\quad where } \Sigma &= \begin{bmatrix} \Lambda \\ 0 \end{bmatrix} \text{ thus } \Sigma^T\Sigma = \Lambda^2
\end{aligned}$$

因而有：
$$\quad X^TX = V\Sigma^T U^T U\Sigma V^T = V\Sigma^T \Sigma V^T  = V \Lambda^2 V^T $$

代入Ridge Regression 结果：

$$\begin{aligned}
(X^TX + \lambda I)^{-1}&= (V \Lambda^2 V^T + \lambda VV^T)^{-1} = V {(\Lambda^2 + \lambda I)} ^{-1}V^T 
\end{aligned}$$
其中${(\Lambda^2 + \lambda I)} $ 是一个对角矩阵，其逆是方便求解的。

故：
$$\begin{aligned}
\theta_{\text{Ridge}} &= (X^TX + \lambda I)^{-1}X^Ty \\ &= V {(\Lambda^2 + \lambda I)} ^{-1}V^T V \Sigma U^T y \\
&= V \left({(\Lambda^2 + \lambda I)} ^{-1} \Sigma \right)\left( U^T y \right) \\
&= V \begin{bmatrix} \frac{\lambda_1}{\lambda_1^2 + \lambda} & 0 & \cdots & 0 \\ 0 & \frac{\lambda_2}{\lambda_2^2 + \lambda} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \frac{\lambda_n}{\lambda_n^2 + \lambda} \end{bmatrix} U^T y \\
&= \sum_{i=1}^n \boxed{\frac{\lambda_i}{\lambda_i^2 + \lambda}} u_i u_i^T y
\\&:= \sum_{i=1}^n \boxed{\frac{\lambda_i}{\lambda_i^2 + \lambda}} ~ \tilde y_i 
\end{aligned}$$

从这里可见：
- $\lambda = 0$ 就相当于原始的OLS，也就是对$y$进行一个坐标变换（SVD）
- $\lambda$很大时，参照Lagrange函数，发现相当于惩罚使得前项（Normal Eqn）并不重要，$\theta_{\text{Ridge}}$趋同
- $\lambda$在正常范围内时，$\lambda_i$则需要纳入考量
  - 这里的$\lambda_i$是数据特征矩阵$X$ SVD的结果，相当于是数据的特征 
  - 而这里$\lambda_i$ 作为SVD的结果，表示的是对应数据的权重，当$\lambda_i$较大时，$\lambda$对于数值的影响就比较小
  - 反之$\lambda_i$较小时，说明这些数据本身对于原训练集的重要性就不那么大（有可能本身这些feature就是一些噪声），而这时的$\lambda$则相对起到了关键作用；
    - *Additionally*，看这里的表达式，当$\lambda_i$较小时，分母上为$\lambda_i^2$（EVEN SMALLER）！所以若在原先的情况下，当存在这样的噪声/干扰情况（$\lambda_i$较小时），就会出现数值不稳定的情况了（即几乎不可逆等）
    - 因此$\lambda$的引入很好的控制了这个内容 
      > ***联想到 Laplace Smoothing***


### LASSO (L1) Regularization, etc.

- **Ridge**

  - 回顾L2 Regularization 的优化目标：
        $$\begin{aligned}
        \min_{\theta} \quad & (X\theta - y)^T(X\theta - y) \\
        \text{s.t.} \quad & \theta^T\theta \leq r
        \end{aligned}$$
        可以发现，其可行域 $\theta^T\theta \leq r$ 为一个圆形的区域；其解集$(X\theta - y)^T(X\theta - y)$为一系列关于$\theta$的二次曲线，如下右图所示

  - 事实上，这里椭圆等高线与圆形可行域相切的点（对应的横纵坐标）即为最优解

- **LASSO** *(Least Absolute Shrinkage and Selection Operator)*: Tibshirani, 1996
  
  - 与Ridge Regression相比，LASSO的惩罚项为L1范数，即：
    $$\begin{aligned}
    \min_{\theta} \quad & \frac1n \sum_{k=1}^n (y_k - \theta^T x_k)^2 \\
    \text{s.t.} \quad & ||\theta||_1 \leq r
    \end{aligned}$$
    
  - 因此其可行域为菱形，而等高线不变；
  - 与Ridge Regression相比，LASSO的解更容易出现在坐标轴附近，因此更加稀疏，即更多的$\theta_i$为0 
  - LASSO对模型起到了 **选择(Selection)** 的作用，而且事实上这种变量的选择是“自动的”


 ![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202308251202029.png)

- **Elastic Net** 
  
  其约束形式为
  $$\alpha ||\theta||_1 + (1-\alpha) ||\theta||_2 \leq r$$

- **Lq Regularization**

  其约束形式为
  $$||\theta||_q \leq r$$


### Weight Decay, CNN - Dropout, etc.

- **Weight Decay**
  
  - 上面提到的 *L1,L2,...*，都是通过限制参数的大小实现其目的，即 Weight Decay
  
  - 而其只是Regularization的一种

- **Dropout**

  - 为了保证模型的泛化能力及其弹性，有时在一定的训练后会随机去掉一定数据，这就是Dropout

  - Dropout相当于一个随机Selection

- **Noise Injection**

  - 在CV中等训练过程中，可以可以注入一定的噪声信息，从而提高模型的鲁棒性

  - 事实上，Ridge Regression也是广义的Noise Injection的一种

## 7.1 Model Selection

**Train / Validation / Test**

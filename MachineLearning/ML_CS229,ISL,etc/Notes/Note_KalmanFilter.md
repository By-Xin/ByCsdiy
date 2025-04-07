# Kalman Filtering

## 简介

### 模型背景

假设有一组数据 $\{X_1, X_2, \ldots, X_n\}$，我们想要估计某一个未知的参数 $\Theta$. 这个参数是未知的, 但是我们假设其在某种意义上取决于数据的分布. 

一般而言, 我们对于这个参数 $\Theta$ 的认知是通过一个系统 $\mathcal{H}$ 来实现的 (可以认为是一个广义的函数). 这个系统 $\mathcal{H}$ 既可能是线性的, 也可能是非线性的. 我们认为这个系统会输入数据 $X(n)$, 并得到输出 $Y(n)$:
$$
X(n) \xrightarrow{\mathcal{H}} Y(n)
$$
我们希望通过对于输入和输出的整理来获得对于系统 $\mathcal{H}$ 的深入的认知. 


### 前 Kalman Filtering 的黑箱系统

在 Kalman Filter 之前, 我们是以一个黑箱的方式来处理这个系统 $\mathcal{H}$. 我们并不知道其中的具体的细节, 只知道输入$X(n)$ 和输出 $Y(n)$. 一般而言, 我们假定输出 $Y(n)$ 是输出 $X(n)$ 的一个函数:
$$
Y(n) = f(X(n), \Theta) 
$$
甚至进一步认为是一个线性函数:
$$
Y(n) = \Theta^\top X(n)
$$
而在这个假设下, 我们可以通过最小二乘法来估计参数 $\Theta$. 这也被称为 Wiener Filter.

### 黑箱系统的局限性与 Kalman Filtering 的核心思想


但是这种黑箱的假设明显过于简单. 
- 事实上在现实生活中, 我们对于这个系统 $\mathcal{H}$ 也并不是完全无知的. 我们可能会有一些来源于其他经验的先验知识. 因此我们希望能够把这些先验知识也加入到我们的模型中 (而不是单纯的一个线性回归). 
- 此外, 并且黑箱的设定还会假设这个系统是不变的. 但是在现实生活中, 系统往往是会随着时间的推移而变化的 (运动是绝对的). 
  
因此我们需要一个动态的模型来描述这个系统 $\mathcal{H}$. 对于任何一个系统, 一定会有内部的发展演化 (Inner Evolution). 而从这种内部演化会部分地反映到人类冰山一角之观测上 (Observation / Measurement). 我们对于这种演化的观测能力是有限的. 这也是人类认知能力的体现与局限. 我们只能通过有限的观测来推测这个系统的演化. 

Kalman Filter 的核心思想也就是在于: **我们能否在有限的观测下, 通过对系统内部演化的建模, 来推测系统的演化?**

### Kalman Filtering 的建模

因此为了对于系统 $\mathcal{H}$ 进行建模, 我们引入了两个概念: State Space (状态空间) 和 Observation Space (观测空间). 对应地, 我们可以用两个方程来描述系统的演化:
$$
\begin{aligned}
&\text{State:} \quad &X_n &= f(X_{n-1}, V_n) \\
&\text{Observation:} \quad &Y_n &= g(X_n, W_n)
\end{aligned}
$$
其中$X_n$ 是系统的状态, $Y_n$ 是观测值, $V_n$ 和 $W_n$ 分别是状态和观测的噪声. 

这样的两个方程就刻画了我们对于系统的动态演化的认知. 
- 我们认为, 每个系统都会有其内部的状态和变化规律. 这些内部状态的维度可能很高, 它们可能是这个系统最本质的刻画 (即 *State*). 
- 但是我们能够看到、真正观测到的, 只是这个系统的状态的某种投影 / 某一方面 / 某个冰山一角 (即 *Observation*). 

这就是我们所说的状态空间和观测空间. 我们希望在有限的观测下, 形成对于状态的认知, 以更好地理解系统的演化. 将状态和观测分离开再结合起来, 是最大的进步. 

此外, 我们对于状态噪声 $V_n$ 和 观测噪声 $W_n$ 的引入也是突破性的. 状态噪声描绘了发展的不确定性, 而观测噪声则描绘了观测的误差. 这两者的引入使得我们对于系统的演化有了更深刻的理解.

## Kalman Filtering 的数学模型

### 模型假设

事实上, 在 Kalman Filtering 中, 我们不需要对平稳性(stationarity) 进行假设. 因此我们可以让这个系统是一个时变的 (即 $f$ 和 $g$ 是时变的, 根据每个时刻 $n$ 来变化):
$$
\begin{aligned}
&\text{State:} \quad &X_n &= f_n(X_{n-1}, V_n) \\
&\text{Observation:} \quad &Y_n &= h_n(X_n, W_n)
\end{aligned}
$$

进一步, 我们依然选择线性化来简化模型 (这样的处理也是合理的. 这在某种意义上相当于 Taylor 展开的一阶近似):
$$
\begin{aligned}
&\text{State:} \quad &X_n &= F_n X_{n-1} + V_n \\
&\text{Observation:} \quad &Y_n &= H_n X_n + W_n
\end{aligned}
$$
其中, $\{X_k\}\in\mathbb{R}^m$ 是系统的状态, $\{Y_k\}\in\mathbb{R}^d$ 是观测值. 二者的维度不同也是合理的. 并且往往我们会有 $m\gg d$. 对应地, $F_n\in\mathbb{R}^{m\times m}$ 和 $H_n\in\mathbb{R}^{d\times m}$ 分别是状态转移矩阵和观测矩阵. 

进一步假设噪声 $V_n$ 和 $W_n$ 零均值的 White Noise:
$$\begin{aligned}
\mathbb{E}[V_n] = \mathbb{E}[W_n] = 0 \\
\text{Cov}[V_n] \triangleq \Sigma_{V_n}\\
\text{Cov}[W_n] \triangleq \Sigma_{W_n} \\
{\text{Cov}}[V_n, W_m] = 0 \quad \forall n,m
\end{aligned}$$

### 两步走策略

对于 Kalman Filtering, 我们希望能够从观测值 $Y_n$ 中来推测系统的状态 $X_n$. 并且当我们如果已经完成了$n-1$时刻的估计, 我们希望 recursively (递归) 地来完成 $n$ 时刻的估计. 具体地, 如同两条腿走路一般, 我们可以分为两个步骤:
1. **预测 (Prediction)**: 通过 $\{Y_1, Y_2, \ldots, Y_{n-1}\}$ 的观测来预测 $\{X_1, X_2, \ldots, X_{n}\}$的状态. 
2. **矫正 (Correction / Update)**: 当我们获得了 $Y_n$ 的观测后, 我们希望能够根据 $\{Y_1, Y_2, \ldots, Y_{n-1}, Y_{n}\}$ 来更新我们对于 $\{X_1, X_2, \ldots, X_{n}\}$ 的估计.

这种两步走的策略体现的是 ***Prediction-Correction*** 的思想. 而这种思想本质上是 **Tracking (跟踪)**.


---

普遍统计学意义上讲, 若给定数据 $\{Y_1, Y_2, \ldots, Y_m\}$, 我们对 $X_n$ 在最小均方误差 (MSE) 准则下的估计就是条件期望:
$$
\hat X_{n|m} = \mathbb{E}[X_n | Y_1, Y_2, \ldots, Y_n] 
$$
这里引入了一个记号: $\hat X_{n|m}$ 表示我们已经掌握的数据是直到 $m$ 时刻的观测值 $\{Y_1, Y_2, \ldots, Y_m\}$ (这也和统计学中条件概率中“给定”的概念类似), 而想要估计的是 $n$ 时刻的$X_n$ 的值. 

上述的两步走的策略也可以用这个记号来表示:
$$
\begin{aligned}
\hat X_{n-1|n-1} \xrightarrow{\text{Prediction}} \hat X_{n|n-1}  \xrightarrow{\text{Correction}} \hat X_{n|n}
\end{aligned}
$$

不过除非是在 Gaussian 分布的情况下, 这个条件期望是很难计算的. 因此我们需要一个更简单的估计方法. 在线性估计中, 引入OLS (最小二乘法) 的思想进行估计. 而 OLS 如果在线性代数的视角下会将其称之为一个投影:
$$
\hat X_{n|m} = \sum_{i=1}^m \hat\alpha_i Y_i = \text{Proj}_{Y_1,\cdots,Y_m} X_n
$$
进行OLS估计(即投影)显然是更方便的. 我们后续也会就此进行详细的讨论.

>***关于投影的理解***
>
> 回顾在回归分析中, 我们想要估计模型 
> $$Y= X\beta + \epsilon$$
> 得到的估计是
> $$
>\hat\beta = (X^\top X)^{-1} X^\top Y
> $$
> 而对$Y$的估计是
> $$
> \hat Y = X\hat\beta = X(X^\top X)^{-1} X^\top Y \triangleq HY
> $$
> 我们有时会将 $H = X(X^\top X)^{-1} X^\top$ 称为“帽子矩阵” (Hat Matrix), 因为 它将 $Y$ 变换为 $\hat Y$. 不过更严谨的讲, 这是一个投影矩阵 (Projection Matrix), 它将 $Y$ 到了一个由 $X$ 张成的子空间上.
> 
> 因此在回归分析中, 我们认为这是 OLS 的结果. 若在线性代数的视角, 我们也可以写作:
> $$
> \hat Y = \text{Proj}_X Y
> $$
> 即
> $\hat Y$ 是 $Y$ 在 $X$ 张成的子空间上的投影. 
> 
> 这里张成的空间(span) 可以理解为是 $X$ 的列空间 (Column Space). 而列空间是一个线性空间, 它是由 $X$ 的列向量线性组合而成的. 在现实意义中列向量就代表一个一个变量/特征, 我们相当于是把一个因变量 $Y$ (其背后可能有错综复杂的自变量关系) 投影到了 (只考虑目前这些) 自变量叫做 $X$ 所组成的线性空间上. 假设 $X$ 的列向量包括 $\mathrm{X}_1, \mathrm{X}_2, \ldots, \mathrm{X}_p$, 那么所谓的$X$的列空间/列向量张成的空间就是所有形如
> $$
> \beta_1 \mathrm{X}_1 + \beta_2 \mathrm{X}_2 + \cdots + \beta_p \mathrm{X}_p
> $$ 
>  的线性组合的集合, 其中 $\beta_1, \beta_2, \ldots, \beta_p$ 是任意实数. 这和我们回归分析的形式是完全一致的.




---

### STEP 1: 预测 (Prediction)

在预测的过程中, 我们需要根据 $\{Y_1, Y_2, \ldots, Y_{n-1}\}$ 来预测 $X_n$, 沿用上面的记号, 我们可以表示为:
$$
\hat X_{n|n-1} = \text{Proj}_{Y_1,\cdots,Y_{n-1}} X_n 
$$
而根据我们之前的假设, 我们对于 $X_n$ 还有这样一个认知:
$$
X_n = F_n X_{n-1} + V_n
$$
因此我们可以将 $X_n$ 的预测值表示为:
$$\begin{aligned}
\hat X_{n|n-1} &= \text{Proj}_{Y_1,\cdots,Y_{n-1}} (F_n X_{n-1} + V_n) \\
&= \text{Proj}_{Y_1,\cdots,Y_{n-1}} (F_n X_{n-1}) + \text{Proj}_{Y_1,\cdots,Y_{n-1}} (V_n) \quad (1) \\
&= \text{Proj}_{Y_1,\cdots,Y_{n-1}} (F_n X_{n-1}) + 0  \quad (2)\\
&= F_n ~\text{Proj}_{Y_1,\cdots,Y_{n-1}} (X_{n-1}) \quad (3)\\
&= F_n ~\hat X_{n-1|n-1} \quad (4)\\
\end{aligned}$$

其中, (1) 是因为投影的线性性, 即 $\text{Proj}(a+b) = \text{Proj}(a) + \text{Proj}(b)$. (2) 是因为 $V_n$ 是一个$n$时刻的噪声, 而知道的观测值是 $\{Y_1, Y_2, \ldots, Y_{n-1}\}$. 未来的噪声和过去的观测是不相关的, 因此 $V_n$ 对于 $\{Y_1, Y_2, \ldots, Y_{n-1}\}$ 的投影是 0. (4) 是由于 $\hat X_{n-1|n-1} = \text{Proj}_{Y_1,\cdots,Y_{n-1}} (X_{n-1})$ 的定义. (3) 的证明较为复杂, 这里不做详细的展开, 但可以粗略地理解为投影的线性性 (这是符合直觉的). 

> **注**: 若对投影这个概念和运算不熟悉, 可以粗略地类比条件期望的运算. 事实上, 条件期望是比投影更严格的概念. 条件期望在数学上相当于某种特殊的投影, 具有投影的所有性质. 但反之不必然成立.

Anyways, 对于预测的结果, 我们可以得到:
$$
\text{(Prediction)}\quad \boxed{\hat X_{n|n-1} = F_n \hat X_{n-1|n-1}}
$$
这一项也被理解为是对于状态的先验估计 (Prior Estimate).

### STEP 2: 矫正 (Correction) / 更新 (Update)

当我们新获得了 $Y_n$ 的观测后, 我们希望能够根据 $\{Y_1, Y_2, \ldots, Y_{n-1}, Y_{n}\}$ 来更新我们对于 $\{X_1, X_2, \ldots, X_{n}\}$ 的估计:
$$\begin{aligned}
\hat X_{n|n} &= \text{Proj}_{Y_1,\cdots,Y_{n}} X_n \\
&= \text{Proj}_{Y_1,\cdots,Y_{n-1}} X_n + \text{Proj}_{\tilde Y_n} X_n\\
&= \hat X_{n|n-1} + \text{Proj}_{\tilde Y_n} X_n 
\quad (\dagger) 
\end{aligned}$$
其中, 
$$\begin{aligned}
\tilde Y_n &= Y_n - \text{Proj}_{Y_1,\cdots,Y_{n-1}} Y_n 
\end{aligned}$$

上述操作背后有更复杂的线性代数原理 (正交补空间) 不作展开, 不过有两个关键的 takeaways:
  - $\text{Proj}_{Y_1,\cdots,Y_{n-1}} Y_n$ 可以理解为 $Y_n$ 对 $Y_1, Y_2, \ldots, Y_{n-1}$ 进行 OLS 线性回归. 因此 $Y_n-\text{Proj}_{Y_1,\cdots,Y_{n-1}} Y_n$ 相当于线性回归后的残差, 其含义类似于 $Y_n$ 在去除了 $Y_1, Y_2, \ldots, Y_{n-1}$ 的所有资讯后独属于 $n$ 时刻的信息.
  - 只有利用独属于 $n$ 时刻独立的信息 $\tilde Y_n$, 上述的最后一个等号才是成立的.

$\diamond$

不论具体形式如何, 对于 ($\dagger$) 式中的 $\text{Proj}_{\tilde Y_n} X_n$, 我们总可以类比 ($X_n$ 关于 $\tilde Y_n$的) 线性回归的模式, 将其表示为 $\tilde Y_n$ 的某种线性组合, 用矩阵可以表示为:
$$
\text{Proj}_{\tilde Y_n} X_n = G_n \tilde Y_n
$$
其中 $G_n$ 是某个投影矩阵, 其具体的形式我们后续会进行讨论. 因此目前我们可以将 ($\dagger$) 式进一步推导为:
$$\begin{aligned}
\hat X_{n|n} &=  \hat X_{n|n-1} + G_n \tilde Y_n \quad (\dagger_2) \\
\end{aligned}$$


$\diamond$

下面具体计算一下 $\tilde Y_n$:
$$\begin{aligned}
\text{Proj}_{Y_1,\cdots,Y_{n-1}} Y_n &= \text{Proj}_{Y_1,\cdots,Y_{n-1}} (H_n X_n + W_n) \quad (1)\\
&= H_n \text{Proj}_{Y_1,\cdots,Y_{n-1}} X_n + \text{Proj}_{Y_1,\cdots,Y_{n-1}} W_n \quad (2)\\
&= H_n \text{Proj}_{Y_1,\cdots,Y_{n-1}} X_n  + 0 \quad (3)\\
&= H_n \hat X_{n|n-1} \quad (4)\\
\end{aligned}$$
其中 (1) 是由 Observation 方程的定义, $Y_n = H_n X_n + W_n$;  (2) 是因为投影的线性性; (3) 中 $\text{Proj}_{Y_1,\cdots,Y_{n-1}} W_n=0$ 同样是因为$n$时刻的噪声 $W_n$ 和过去的观测值是独立的; (4) 是也是由 STEP 1 中的定义: $\hat X_{n|n-1} = \text{Proj}_{Y_1,\cdots,Y_{n-1}} X_n$ 得到的.

故我们可以得到:
$$\begin{aligned}
\tilde Y_n &= Y_n - \text{Proj}_{Y_1,\cdots,Y_{n-1}} Y_n \\
&= Y_n - H_n \hat X_{n|n-1} 
\end{aligned}$$

$\diamond$

在算得 $\tilde Y_n$ 的具体形式后, 我们可以将其代入 ($\dagger_2$) 式中:
$$\begin{aligned}\text{(Correction)} \quad
\boxed{
\hat X_{n|n} =  \hat X_{n|n-1} + G_n (Y_n - H_n \hat X_{n|n-1})}
\end{aligned}$$

这是 Kalman Filtering 的核心公式.  这一步作为 Kalman Filtering 的更新/矫正步骤, 其核心思想是: **我们通过对 $Y_n$ 的观测来更新我们对于 $X_n$ 的估计.** 
- $H_n \hat X_{n|n-1}$ 相当于当我们只有 $n-1$ 时刻的信息时, 我们用对 $n$ 时刻系统状态的估计 $\hat X_{n|n-1}$ 来预测 $n$ 时刻观测的值.
- ${Y_n - H_n \hat X_{n|n-1}}$ 则反映了当我们真的有了 $Y_n$ 的观测后, 我们刚刚的预测相对于真实观测的偏差. 若这个值的绝对值很大 (偏差很大), 则说明我们本身的预测是有问题的, 我们对于 $\hat X_{n|n-1}$ 的估计是有问题的, 需要大幅矫正更新. 这一项也被称为 residual / error / **innovation** (新息). 由于 innovation 代表了新观测中包含的"新信息"，即不能从先前估计中预测出的部分. 因此在理想情况下, 若 innovation 是一个 white noise, 则说明这个滤波器是良好运行的, 新的观测和我们的预测是完全一致的.
- 因此我们就根据我们的偏差大小来更新我们的估计 $\hat X_{n|n-1}$, 而 $G_n$ 则相当于调整我们更新的幅度 (在 Filtering 中, $G_n$ 也被称为 **Kalman Gain**. 其也类似于机器学习领域中的学习率). 因此 $G_n (Y_n - H_n \hat X_{n|n-1})$ 就是我们对于 $\hat X_{n|n-1}$ 的更新量.

---

上述的 Prediction 和 Correction 的过程就是 Kalman Filtering 最核心的思想. 
$$
\text{Kalman Filtering} \quad
\begin{aligned}
&\text{Prediction:} \quad \hat X_{n|n-1} = F_n \hat X_{n-1|n-1} \\
&\text{Correction:} \quad \hat X_{n|n} =  \hat X_{n|n-1} + G_n (Y_n - H_n \hat X_{n|n-1})
\end{aligned}
$$

### Kalman Gain 的选取

在 Kalman Filtering 中也存在一个关于 **sensitivity** 和 **stability** 的 trade-off 问题:
- 当 $G_n$ 较小时, 其整体的稳定性较好 (对于噪音的鲁棒性较强), 但是对于系统的变化的敏感度较低 (即对系统的变化反应较慢).
- 当 $G_n$ 较大时, 其调整的幅度较大, 敏感度较高, 但是对于噪音的鲁棒性较差 (即对系统的变化反应较快).

因此对于 $G_n$ 的选取也很重要. 经过推导, 我们可以得到最优的 $G_n$ 的选取:
$$
\boxed{G_n = H_n^\top \hat R_{n|n-1} (H_n \hat R_{n|n-1} H_n^\top + \Sigma_{W_n})^{-1}}
$$
其中 $\hat R_{n|n-1} = \text{Cov}[X_n - \hat X_{n|n-1}]$ 是预测误差 $X_n - \hat X_{n|n-1}$ 的协方差矩阵. 

### 预测误差协方差 (Prediction Error Covariance) 的计算

这里的一个问题是 $\hat R_{n|n-1}$ 的计算也并不是直接得到的. 但是根据 Kalman 的思想, 这个误差本身也可以看作是一个状态, 因此我们可以将其也通过两步走的策略进行递推:
$$
\begin{aligned}
\hat R_{n|n-1} \longrightarrow \hat R_{n|n-1} \longrightarrow \hat R_{n|n}
\end{aligned}
$$

具体地, 根据定义:
$$\begin{aligned}
\hat R_{n|n-1} &= \text{Cov}[X_n - \hat X_{n|n-1}] \\
&= \mathbb{E}[(X_n - \hat X_{n|n-1})(X_n - \hat X_{n|n-1})^\top] \\
&= \mathbb{E}(F_n X_{n-1} + V_n - \hat X_{n|n-1})(F_n X_{n-1} + V_n - \hat X_{n|n-1})^\top \quad (1)\\
&= \mathbb{E}(F_n X_{n-1} + V_n  - F_n \hat X_{n-1|n-1})(F_n X_{n-1} + V_n - \hat X_{n|n-1})^\top \quad (2)\\
&= \mathbb{E}[F_n (\underbrace{X_{n-1} - \hat X_{n-1|n-1}}_{\text{Filtering Error}})+ V_n] [F_n (X_{n-1} - \hat X_{n-1|n-1})+ V_n]^\top \\
&= F_n \mathbb{E}[(X_{n-1} - \hat X_{n-1|n-1})(X_{n-1} - \hat X_{n-1|n-1})^\top] F_n^\top + \mathbb{E}(V_n V_n^\top) \quad (3)\\
&= F_n \hat R_{n-1|n-1} F_n^\top + \Sigma_{V_n} \quad (4)\\
\end{aligned}$$
其中, (1) 是根据预测方程的定义 $X_n = F_n X_{n-1} + V_n$; (2) 是根据 STEP 1 Prediction 推导的结果 $\hat X_{n|n-1} = F_n \hat X_{n-1|n-1}$; (3) 是根据噪声的独立性; (4) 是根据协方差的定义: $\hat R_{n-1|n-1} = \text{Cov}[X_{n-1} - \hat X_{n-1|n-1}]$.

最终我们可以得到:
$$
\text{(Prediction Error Covariance)} \quad
\boxed{\hat R_{n|n-1} = F_n \hat R_{n-1|n-1} F_n^\top + \Sigma_{V_n}}
$$

### 滤波误差协方差 (Filtering Error Covariance) 的计算

当我们给出了 Prediction Error Covariance 的计算后, 其计算还依赖于 $\hat R_{n-1|n-1}$ (flitering error covariance) 的计算. 下面给出该项的计算方法.

$$\begin{aligned}
\hat R_{n|n} &= \text{Cov}[X_n - \hat X_{n|n}] \\
&= \mathbb{E}(X_n - \hat X_{n|n})(X_n - \hat X_{n|n})^\top \\
&= \mathbb{E}[X_n - \hat X_{n|n-1} - G_n (Y_n - H_n \hat X_{n|n-1})][\sim ]^\top \quad (1)\\
&= \mathbb{E}[X_n  - \hat X_{n|n-1} - G_n H_n (X_n - \hat X_{n|n-1}) - G_n W_n] [\sim ]^\top \quad (2)\\
&= \mathbb{E}[ (I - G_n H_n) (X_n - \hat X_{n|n-1}) - G_n W_n] [\sim ]^\top \quad (3)\\
&= (I - G_n H_n) \mathbb{E}[(X_n - \hat X_{n|n-1})(X_n - \hat X_{n|n-1})^\top] (I - G_n H_n)^\top+ G_n \Sigma_{W_n} G_n^\top \quad (4)\\
&= (I - G_n H_n) \hat R_{n|n-1} (I - G_n H_n)^\top + G_n \Sigma_{W_n} G_n^\top \quad (5)\\
\end{aligned}$$
其中, $\sim$ 代表和前面的式子相同的部分, 这里不再重复书写; (1) 是根据 STEP 2 Correction 的定义: $\hat X_{n|n} = \hat X_{n|n-1} + G_n (Y_n - H_n \hat X_{n|n-1})$; (2) 是根据 Observation 方程的定义 $Y_n = H_n X_n + W_n$; (3) (4) 是单纯的线性代数运算; (5) 是根据协方差的定义: $\hat R_{n|n-1} = \text{Cov}[X_n - \hat X_{n|n-1}]$.

因此我们可以得到:
$$
\text{(Filtering Error Covariance)} \quad
\boxed{\hat R_{n|n} = (I - G_n H_n) \hat R_{n|n-1} (I - G_n H_n)^\top + G_n \Sigma_{W_n} G_n^\top}
$$
其中, $I$ 是单位矩阵.


### Kalman Filtering 的总结

综上, 我们就得到了 Kalman Filtering 的完整的公式:
$$
\begin{aligned}
&\text{Prediction:} \quad \hat X_{n|n-1} = F_n \hat X_{n-1|n-1} \\
&\text{Correction:} \quad \hat X_{n|n} =  \hat X_{n|n-1} + G_n (Y_n - H_n \hat X_{n|n-1}) \\
&\text{Kalman Gain:} \quad G_n = H_n^\top \hat R_{n|n-1} (H_n \hat R_{n|n-1} H_n^\top + \Sigma_{W_n})^{-1} \\
&\text{Prediction Error Covariance:} \quad \hat R_{n|n-1} = F_n \hat R_{n-1|n-1} F_n^\top + \Sigma_{V_n} \\
&\text{Filtering Error Covariance:} \quad \hat R_{n|n} = (I - G_n H_n) \hat R_{n|n-1} (I - G_n H_n)^\top + G_n \Sigma_{W_n} G_n^\top
\end{aligned}
$$
# Time Series Decomposition

## Introduction

一个时间序列经验上通常具有三个主要组成部分：趋势 (Trend), 季节性 (Seasonality) 和噪声 (Noise). 广义而言, 时间序列可以表示为这三个部分的某种组合:
$$
x_t = F(T_t, S_t, N_t)
$$
其中:
- $T_t$ 为趋势项, 通常认为其增长较为缓慢
- $S_t$ 为季节性项, 通常认为其周期性变化
- $N_t$ 为噪声项, 通常认为其是随机的, 包含了所有未被趋势和季节性解释的部分

若能很好的对时间序列进行分解，就可以更好的理解时间序列的特性. 有时我们也会将一个时间序列去除其中的趋势和季节性进行分析建模. 通常称这种分解为 detrending 和 seasonal adjustment.

## Classical Decomposition (Additive Model)

### Additive Model 概念与假设

一个最简单的时间序列分解模型是加法模型 (Additive Model). 在加法模型中, 时间序列被分解为上述三个部分的简单加和:
$$
x_t = T_t + S_t + N_t
$$
其中, 
-  $R_t$ 为随机项, 通常认为是一个 $0$ 均值的 stochastic process, 即对任意 $m$ 个不同的时间点 $i_1, i_2, \ldots, i_m$, 及其对应的随机项 $R_{i_1}, R_{i_2}, \ldots, R_{i_m}$, 有:
    $$
    \frac{1}{m} \sum_{j=1}^m R_{i_j} \stackrel{p}{\to} 0, \quad m \to \infty
    $$
- $S_t$ 为季节性项, 通常认为其是一个周期性变化的函数, 且周期为 $p$:
    $$
    S_{t+p} = S_t, \quad \forall t
    $$
  - 通常进一步假设在一个周期内(如 $S_t, S_{t+1}, \ldots, S_{t+p-1}$)的均值为 $0$:
    $$
    \frac{1}{p} \sum_{j=0}^{p-1} S_{t+j} = 0, \quad \forall t
    $$  

---

### Trend Estimation

在上述假设基础上, 我们可以通过简单的平均法来估计趋势项 $T_t$: 取一个长度为 $k+l+1$ 期的离散时间序列 $\left\{x_{t-k}, x_{t-k+1}, \ldots, x_t, x_{t+1}, \ldots, x_{t+l}\right\}$, 且使得 $k+l+1$ 恰好为一个周期的整数倍, 即 $k+l+1 = n \cdot p$, 则这**整数段周期内的时序数据之平均值即为趋势项的估计.** 这是因为:
$$
\frac{1}{m} \sum_{j=-k}^l x_{t+j} \approx \frac{1}{m}  \sum_{j=-k}^l \left(T_{t+j} + S_{t+j} + N_{t+j}\right) = T_t + \frac{1}{m} \sum_{j=-k}^l S_{t+j} + \frac{1}{m} \sum_{j=-k}^l N_{t+j} \approx T_t
$$
- $T_t$ 项其变化较为缓慢, 故在该时间段内可认为变化忽略不计. 故其均值即为 $T_t$
- $S_t$ 项是由于假设该时间段内恰包含了整数个周期, 故其均值为 $0$
- $R_t$ 项是随机项, 故其均值为 $0$

### Seasonality Estimation

在估计出趋势项 $T_t$ 后, 我们可以用 $x_t - T_t$ 来去除趋势项的影响, 从而估计出季节性项 $S_t$. 简单而言, 我们**对去除趋势项之后的数据, 在周期内相同位置的数据取平均值即可估计出该位置的季节性成分$S_t$.** 原因如下. 

若记周期长度为 $p$ (如一年有12个月则 $p=12$), 在一个周期内的位置为 $k$ (如第 $k$ 个月), 跨越周期的数量为 $j = 0, 1, \ldots, m$ (如 第 $j$ 年), 则:
$$
\frac{1}{m} \sum_{j=1}^{m}(x_{jp+k} - T_{jp+k}) \approx \frac{1}{m} \sum_{j=1}^{m} S_{jp+k} + \frac{1}{m} \sum_{j=1}^{m} R_{jp+k} \approx S_k
$$
- $S_{jp+k}$ 项由于假设$S_t$是随着周期重复的, 因此对于同一位置的数据, 其季节性成分是相同的, 均值即为 $S_k$
- $R_{jp+k}$ 项是随机项, 故其均值渐进为 $0$

在分别估计出趋势项 $\hat{T}_t$ 和季节性项 $\hat{S}_t$ 后, 我们可以通过 $x_t - \hat{T}_t - \hat{S}_t$ 来估计出噪声项 $\hat{R}_t = x_t - \hat{T}_t - \hat{S}_t$


## Multiplicative Model

有时时间序列的趋势和季节性效应并不是通过加法进行作用的. 另一种常见的模型是乘法模型 (Multiplicative Model):
$$
x_t = T_t \cdot S_t \cdot R_t
$$
但这可以看作是对原始时序数据进行对数变换后的加法模型. 根据对数性质有:
$$
\log(x_t) = \log(T_t) + \log(S_t) + \log(R_t)
$$
其后续的分解过程与加法模型类似.

## Advanced Decomposition

上面的 Classical Decomposition 具有一些局限性, 如:
- 季节性保持不变的假设可能不合理
- 对于异常值过于敏感

Cleveland 等人提出了一种更为灵活的分解方法, 即 STL (Seasonal and Trend decomposition using Loess).  该方法通过局部加权回归 (Locally Weighted Regression) 来估计趋势和季节性, 从而更好的适应时间序列的变化. STL允许季节性随时间变化, 且对异常值更为 robust. 

在 R 中, 可以通过调用类似
```r
aus_arrivals |>
    filter(Origin == "Japan") |>
    model(
        STL(Arrivals ~ trend() + season())) |>
    components() |>
    autoplot()
```
的代码来进行 STL 分解.
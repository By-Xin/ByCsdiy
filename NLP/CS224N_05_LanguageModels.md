# Language Models

## Introduction

直观上的理解, Language Model是一个用来预测下一个词的模型. 形式上, 若给定一个句子$w_1, w_2, ..., w_{i-1}$, 则Language Model的目标是预测下一个词 $w_i$ 应该是 $\mathcal{V} = \{w_1, w_2, ..., w_{|V|}\}$ 中的哪一个. 这种选择可以建模为在给定 $w_1, w_2, ..., w_{i-1}$ 的条件下, $w_i$ 的概率分布, 即:
$$
\mathbb{P}(w_i | w_1, w_2, ..., w_{i-1})
$$
而更广义的讲, Language Model 的目标是从 0 开始预测整个句子的概率分布, 即尝试预测出整个序列 $w_1, w_2, ..., w_m$, 用数学语言描述为其联合概率分布:
$$\begin{aligned}
\mathbb{P}(w_1, w_2, \cdots  w_m) &= \mathbb{P}(w_1) \mathbb{P}(w_2 | w_1) \mathbb{P}(w_3 | w_1, w_2) \cdots \mathbb{P}(w_m | w_1, w_2, \cdots, w_{m-1}) \\
&= \prod_{i=1}^{m} \mathbb{P}(w_i | w_1, w_2, \cdots,  w_{i-1})
\end{aligned}$$

Language Model 的研究就是在于如何精准地估计这些概率分布. 

## N-gram Language Model

### N-gram Model 理论

在深度学习方法流行之前, 一种常见的方法是使用 n-gram 模型来估计这个概率分布
$$\begin{aligned}
\mathbb{P}(w_1, w_2, \cdots  w_m) 
&= \prod_{i=1}^{m} \mathbb{P}(w_i | w_1, w_2, \cdots,  w_{i-1})
\end{aligned}$$

对于 n-gram 模型, 一个重要假设是 Markov 假设, 即假设当前词的预测只依赖于前面的最接近的 $n$ 个词 (而不是需要依赖整个历史). 这样, 上述概率分布可以近似为:
$$\begin{aligned}
\mathbb{P}(w_1, w_2, \cdots  w_m)
&\approx \prod_{i=1}^{m} \mathbb{P}(w_i | w_{i-n+1}, w_{i-n+2}, \cdots,  w_{i-1})
\end{aligned}$$

注意, n-gram 本身指的是长度为 $n$ 个词的语段, 在 Markov 条件下, 我们实际上是用前 $n-1$ 个词来预测第 $n$ 个词, 即尝试计算概率: 
$$\begin{aligned}
\mathbb{P}(w_{t+1} | w_{t}, w_{t-1}, \cdots,  w_{t-n+2})   = \frac{\mathbb{P}(w_{t+1}, w_{t}, w_{t-1}, \cdots,  w_{t-n+2})}{\mathbb{P}(w_{t}, w_{t-1}, \cdots,  w_{t-n+2})}
\end{aligned}$$

对右侧的概率分布的一种常见估计就是用频率代替概率, 即:
$$\begin{aligned}
\mathbb{P}(w_{t+1} | w_{t}, w_{t-1}, \cdots,  w_{t-n+2})   \approx  \frac{ \text{count}(w_{t+1}, w_{t}, w_{t-1}, \cdots,  w_{t-n+2})}{\text{count}(w_{t}, w_{t-1}, \cdots,  w_{t-n+2})}
\end{aligned}$$
其本质上就是在统计语料库中某种语段出现的频率, 用最频繁出现的语段来作为预测的依据. 

当然在求出其概率分布后, 我们既可以固定选择最大概率的词, 也可以以这个概率作为权重来进行随机抽样, 增加模型的多样性.

然而在实践中, 我们经常会遇到一个$n$ 选择的权衡问题. 例如考虑句子: *As the proctor started the clock, the students opened their ( )"*. 若 $n$ 长度不足, 则很有可能忽略掉关键的信息 *proctor*. 但是另一方面, 若 $n$ 太长, 则很有可能会遇到数据稀疏的问题, 例如我们很难在语料库中找到连续 $20$ 个词都相同的语段. 

因此 n-gram 模型经常会面临两个问题:
- **Sparsity problem**: 数据稀疏问题, 即很多语段在语料库中没有出现过, 从而导致概率分布的估计不准确. 通常这种 sparsity 的问题是由于 $n$ 过大造成的. 经验而讲, 适宜的范围为 $n\leq 5$. Sparsity problem 具体有两种表现方式. 考虑一个简单的 3-gram 模型的概率估计 $\text{count}(w_{1}, w_{2}, w_{3}) / \text{count}(w_{1}, w_{2})$, 
    - 若 $w_1, w_2, w_3$ 从未一起在语料库中出现过, 则 $\text{count}(w_{1}, w_{2}, w_{3}) = 0$, 从而导致概率估计为 $0$. 一种直白的解决方法是使用 Laplace Smoothing, 即在分子分母上都加上一个小量 $\delta$, 从而避免概率为 $0$ 的情况.
    - 若 $w_1, w_{2}$ 组合都没有在语料库中出现过, 则 $\text{count}(w_{1}, w_{2}) = 0$, 从而导致概率估计的分母为 $0$, 这甚至直接导致了整个概率分布的计算不可行. 一种解决方法是使用 Back-off, 降低 $n$ 的长度, 即当 $w_1, w_{2}$ 组合没有出现过时, 退而求其次, 使用 $w_{2}$ 单独出现的概率来代替.
- Storage problem: 由于 n-gram 模型需要存储这个语料库中的所有 n-gram 语段, 因此当 $n$ 较大或语料库的规模较大时, 会导致存储空间的问题.

### N-gram Model Python 实现

```python



## Window-based Neural Language Model
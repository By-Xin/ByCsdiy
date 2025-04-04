# From Transformer to Mamba

> Ref: 李宏毅, 生成式AI時代下的機器學習(2025) https://youtu.be/gjsdVi90yQo?si=fNjhGg7DtaOy09ZU

## Every Architecture Has Its Own Reason to Exist

每一种架构都有一个存在的理由.  

对于 RNN, Self-Attention, Mamba, etc. 这一系列的架构, 其核心都是在处理与序列相关的任务.  对于输入 $x_1, x_2, \cdots, x_t$, 对应输出 $f(x_1, x_2, \cdots, x_t) = (y_1, y_2, \cdots, y_t)$. 其中又通常会有因果性的要求, 即输出 $y_t$ 只能依赖于 $t$ 及其之前的资讯. 

### RNN-Style Architecture

对于循环神经网络类, 其核心在于维护一个 hidden state $\mathrm{H}_t$ 作为记忆, 其更新方式为:
$$
\mathrm{H}_t = f_{A,t}(\mathrm{H}_{t-1})+f_{B,t}(\boldsymbol{x}_t)
$$
再通过另一个函数 $f_{C,t}$ 进行输出:
$$
\boldsymbol{y}_t = f_{C,t}(\mathrm{H}_t)
$$
其中 $f_{A,t},f_{B,t},f_{C,t}$ 都是神经网络, 并且可以根据时间 $t$ (输入序列的位置) 的不同而不同. 例如 LSTM 中, $f_{B,t}$ 对应 input gate, $f_{A,t}$ 对应 forget gate, $f_{C,t}$ 对应 output gate.  

![RNN-Style Architecture](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250403125423.png) 

### Self-Attention

![Self-Attention Style Architecture](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250403130039.png)

Self-Attention 与 RNN 在 inference 时的区别在于: 对于 RNN, 其每一步的运算量都是恒定的. 但是对于 Self-Attention, 其运算量是随着输入序列的长度增加而增加的. 并且对于 RNN, 当我们需要 $y_t$ 的输出时, 我们只需要 $\mathrm{H}_{t-1}, x_t$ 即可. 但是对于 Self-Attention, 我们需要 $x_1, x_2, \cdots, x_t$ 的所有信息才能计算 $y_t$. 这就导致了 Self-Attention 的内存开销也是随着输入序列的长度增加而增加的. 

而另一方面, 在 training 时, Self-Attention 的优点在于可以并行化. 这主要是通过 Shifted Input 和 Masked Attention 来实现通过并行的方式模拟序列的输入输出. 此外其架构的设计更好的利用了 GPU 的并行计算能力. 这也是为什么近年 Transformer 取代 RNN 成为主流的原因.

### From Transformer to Mamba

随着大语言模型的不断发展, 我们输入的 input 的长度也越来越长. 这就导致了 Self-Attention 的内存开销也越来越大, 而 RNN 在 inference 时的优势也越来越明显. 因此这就抛出了一个问题: *RNN 真的没有办法并行化吗?*

#### RNN 的并行化设计

回顾其架构:
$$\begin{align*}
\mathrm{H}_1 &= f_{A,1}(\mathrm{H}_0)+f_{B,1}(\boldsymbol{x}_1) \\
\mathrm{H}_2 &= f_{A,2}(\mathrm{H}_1)+f_{B,2}(\boldsymbol{x}_2) \\
\vdots \\
\mathrm{H}_t &= f_{A,t}(\mathrm{H}_{t-1})+f_{B,t}(\boldsymbol{x}_t) \\
\boldsymbol{y}_t &= f_{C,t}(\mathrm{H}_t)
\end{align*}$$

假设 $f_{A,1}(\mathrm{H}_0) = 0$, 则上面的 RNN 架构可以展开为:
$$\begin{align*}
\mathrm{H}_1 &= f_{B,1}(\boldsymbol{x}_1) \\
\mathrm{H}_2 &= f_{A,2}(f_{B,1}(\boldsymbol{x}_1)) + f_{B,2}(\boldsymbol{x}_2) \\
\mathrm{H}_3 &= f_{A,3}(f_{A,2}(f_{B,1}(\boldsymbol{x}_1)) + f_{B,2}(\boldsymbol{x}_2)) + f_{B,3}(\boldsymbol{x}_3) \\
\vdots \\
\mathrm{H}_t &= f_{A,t}(f_{A,t-1}(\cdots f_{A,2}(f_{B,1}(\boldsymbol{x}_1)) + f_{B,2}(\boldsymbol{x}_2) \cdots)) + f_{B,t}(\boldsymbol{x}_t) \\
\boldsymbol{y}_t &= f_{C,t}(\mathrm{H}_t)
\end{align*}$$

而在上述架构中, 最耗费时间的部分是一系列嵌套的 $f_{A,t}$ 的计算. 一个解决方法是令 $f_{A,t}$ 为 identity function (即 $f_{A,t}(\mathrm{H}) = \mathrm{H}$), 这样就可以将其展开为:
$$\begin{align*}
\mathrm{H}_1 &= f_{B,1}(\boldsymbol{x}_1) \\
\mathrm{H}_2 &= \mathrm{H}_1 + f_{B,2}(\boldsymbol{x}_2) = f_{B,1}(\boldsymbol{x}_1) + f_{B,2}(\boldsymbol{x}_2) \\
\mathrm{H}_3 &= \mathrm{H}_2 + f_{B,3}(\boldsymbol{x}_3) = f_{B,1}(\boldsymbol{x}_1) + f_{B,2}(\boldsymbol{x}_2) + f_{B,3}(\boldsymbol{x}_3) \\
\vdots \\
\mathrm{H}_t &= \mathrm{H}_{t-1} + f_{B,t}(\boldsymbol{x}_t) = f_{B,1}(\boldsymbol{x}_1) + f_{B,2}(\boldsymbol{x}_2) + \cdots + f_{B,t}(\boldsymbol{x}_t) \\
\end{align*}$$

进一步作如下假设: 假设 $\mathrm{H}_t \in \mathbb{R}^{d\times d}$, $f_{B,t}(\boldsymbol{x}_t) \triangleq D_t \in \mathbb{R}^{d\times d}$, $f_{C,t}(\mathrm{H}_t) := \mathrm{H}_t \boldsymbol{q}_t$, 其中 $\boldsymbol{q}_t = W_Q \boldsymbol{x}_t$.  则此时可以将其展开为:
$$\begin{align*}
\mathrm{H}_1 &=  D_1 \\
\mathrm{H}_2 &= D_1 + D_2 \\
\vdots \\
\mathrm{H}_t &= D_1 + D_2 + \cdots + D_t \\
\end{align*}$$
且
$$\begin{align*}
\boldsymbol{y}_1 &= D_1 \boldsymbol{q}_1 \\
\boldsymbol{y}_2 &= (D_1 + D_2) \boldsymbol{q}_2 \\
\cdots \\
\boldsymbol{y}_t &= (D_1 + D_2 + \cdots + D_t) \boldsymbol{q}_t
\end{align*}$$

若再假设 $D_t := \boldsymbol{v}_t \boldsymbol{k}_t^\top$,  其中 $\boldsymbol{v}_t:=W_v \boldsymbol{x}_t$, $\boldsymbol{k}_t := W_k \boldsymbol{x}_t$, 则可以将其展开为:
$$\begin{align*}
\boldsymbol{y}_1 &= \boldsymbol{v}_1 \boldsymbol{k}_1^\top \boldsymbol{q}_1 \\
\boldsymbol{y}_2 &= \boldsymbol{v}_1 \boldsymbol{k}_1^\top \boldsymbol{q}_2 + \boldsymbol{v}_2 \boldsymbol{k}_2^\top \boldsymbol{q}_2 \\
\boldsymbol{y}_3 &= \boldsymbol{v}_1 \boldsymbol{k}_1^\top \boldsymbol{q}_3 + \boldsymbol{v}_2 \boldsymbol{k}_2^\top \boldsymbol{q}_3 + \boldsymbol{v}_3 \boldsymbol{k}_3^\top \boldsymbol{q}_3 \\
\vdots \\
\boldsymbol{y}_t &= \boldsymbol{v}_1 (\boldsymbol{k}_1^\top \boldsymbol{q}_t )+ \boldsymbol{v}_2 (\boldsymbol{k}_2^\top \boldsymbol{q}_t) + \cdots + \boldsymbol{v}_t (\boldsymbol{k}_t^\top \boldsymbol{q}_t) := \sum_{j=1}^t \alpha_{t,j} \boldsymbol{v}_j
\end{align*}$$

这正是 Self-attention 的形式! 唯一的不同就是没有进行 Softmax. 这里称之为 Linear Attention. 

因此我们现在有如下认知:
- Linear Attention 就是广义 RNN 去掉了 Reflection $f_{A,t}$ 
- Linear Attention 就是 Self-Attention 去掉了 Softmax 

因此我们就可以在 Inference 时将其理解为一个 RNN, 但是在 Training 时当作(没有Softmax 的) Self-Attention 来训练. 这样就可以在 Inference 时达到 RNN 的效果, 但是在 Training 时又可以利用 Self-Attention 的并行化优势.

#### Linear Attention 

> *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (https://arxiv.org/abs/2006.16236)*

这里梳理一下 Linear Attention 的计算过程:
$$\begin{align*}
\mathrm{H}_t = \mathrm{H}_{t-1} + f_{B,t}(\boldsymbol{x}_t) &,\quad 
f_{B,t}(\boldsymbol{x}_t) = \boldsymbol{v}_t \boldsymbol{k}_t^\top \\
\boldsymbol{y}_t = f_{C,t}(\mathrm{H}_t) &,\quad
f_{C,t}(\mathrm{H}_t) = \mathrm{H}_t \boldsymbol{q}_t
\end{align*}$$
其中 $\boldsymbol{v}_t = W_v \boldsymbol{x}_t\in\mathbb{R}^{d'}, \boldsymbol{k}_t = W_k \boldsymbol{x}_t\in\mathbb{R}^{d}, \boldsymbol{q}_t = W_q \boldsymbol{x}_t\in\mathbb{R}^{d}$. 这里 $W_v, W_k, W_q$ 都是可学习的参数.

因此对于 hidden state $\mathrm{H}_t$, 其更新的方式为:
$$
\mathrm{H}_t = \mathrm{H}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top 
=\mathrm{H}_{t-1} + \begin{bmatrix}
k_{t,1} \boldsymbol{v}_{t}~; & k_{t,2} \boldsymbol{v}_{t} ~;& \cdots ~;& k_{t,d} \boldsymbol{v}_{t}
\end{bmatrix}
$$
直观的理解为, $\boldsymbol{v}_t$ 是希望在$t$时刻更新到 hidden state 的信息, 而 $\boldsymbol{k}_t$ 决定了这笔资讯加入 hidden state 的位置. 例如, 一个极端例子若 $\boldsymbol{k}_t = \begin{bmatrix} 1 & 0 & \cdots & 0 \end{bmatrix}^\top$, 则 $\boldsymbol{v}_t$ 只会更新到 $\mathrm{H}_t$ 的第一维, 而其他维度不变. 当然在更一般的情况下可以分散到多个维度上.

而最终的输出 $\boldsymbol{y}_t$ 为: 
$$
\boldsymbol{y}_t = \mathrm{H}_t \boldsymbol{q}_t
$$

这在直观上可以理解为, $\mathrm{H}_t$ 是我们维护到目前为止的所有信息, 不同的维就对应着不同的信息. 而 $\boldsymbol{q}_t$ 则决定我们从每个维度要提取多少的信息. 若同样的 $\boldsymbol{q}_t = \begin{bmatrix} 1 & 0 & \cdots & 0 \end{bmatrix}^\top$, 则 $\boldsymbol{y}_t$ 只会提取到 $\mathrm{H}_t$ 的第一维. 

#### RNN 与 Transformer 的记忆

一个普遍的认知是, 由于 hidden state $\mathrm{H}_t\in\mathbb{R}^{d,d}$, 因此其记忆的容量是有限的. 而 Transformer 由于其可以一直读取到整个序列, 因此其记忆的容量看似是无限的. 但是事实并非如此. 

在 Transformer (Self-Attention + Softmax) 中, 每个输入 $x_1, x_2, \cdots, x_t$ 都会对应 Q, K, V: $\boldsymbol{q}_t, \boldsymbol{k}_t \in \mathbb{R}^{d}$, $\boldsymbol{v}_t \in \mathbb{R}^{d'}$. 当当前序列的长度 $t\leq d$ 时, 我们总能设计出一组合理的 $\boldsymbol{k}_t$ 使得每一笔资讯都能被良好存储.  然而当 $t>d$ 时, 对于一个 $d$ 维向量, 我们最多只能找到 $d$ 个正交的向量, 而对于其余的向量, 我们总能将其表示为其他向量的线性组合. 这也就导致了即使我们只想提取某个时刻 $t_0$ 的资讯, 但是由于 $t$ 的总长度过长, 也总会存在其他的杂讯 $t'$ 使得 $\alpha_{t,t'}$ 的值过大, 引入了我们并不需要的 value $\boldsymbol{v}_{t'}$. 这也就导致了 Transformer 的记忆容量是有限的.

因此, 在实践中 RNN 的效果不如 Transformer 的原因可能并不是因为其记忆容量有限, 而是因为 Linear Attention 的设计中 $\mathrm{H}_t$ 对已有的输入 $\mathrm{H}_{t-1}$ 不会遗忘, 已有的记忆不会再发生改变.  而在 Transformer  (Self-Attention + Softmax) 中, 正是因为有了 Softmax 的设计, 使得每次有了新的资讯的时候, 全部的已有信息的重要性 (attention score) 都会被重新加权计算, 这也就给了 Transformer 更大的灵活性和对历史更新的能力.

#### 可遗忘的 Linear Attention

***RetNet***

鉴于 Linear Attention 的一个弊病就是无法遗忘已有的记忆, 因此我们可以设计一个 Retention Network 来解决这个问题:
$$\begin{align*}
\mathrm{H}_t &= \gamma \mathrm{H}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top \\
\boldsymbol{y}_t &= \mathrm{H}_t \boldsymbol{q}_t
\end{align*}$$
其中 $\gamma \in [0,1]$ 是一个衰减参数. 

***Gated Retention***

甚至更进一步, 让这个衰减参数 $\gamma$ 也可以是时变的:
$$\begin{align*}
\mathrm{H}_t &= \gamma_t \mathrm{H}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top \\
\boldsymbol{y}_t &= \mathrm{H}_t \boldsymbol{q}_t
\end{align*}$$
其中 $\gamma_t = \text{sigmoid}(W_\gamma \boldsymbol{x}_t)$, $W_\gamma$ 是可学习的参数. 这样就可以在每个时刻都可以选择是否要遗忘已有的记忆.

***更复杂的 Reflection***

甚至可以设计一个更复杂的 Reflection:
$$\begin{align*}
\mathrm{H}_t &= G_t \odot (\mathrm{H}_{t-1}) + \boldsymbol{v}_t \boldsymbol{k}_t^\top \\
G_t &= \boldsymbol{e}_t \boldsymbol{s}_t^\top
\end{align*}$$
其中 $\boldsymbol{e}_t, \boldsymbol{s}_t$ 是可学习的. 

类似的设计还有很多, 而 Mamba 也是其中之一.

![类似的可以看作 RNN 的架构还有很多 arxiv.org/abs/2406.06484](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250403145422.png)

#### Delta Net

> *DeltaNet: Parallelizing Linear Transformers with the Delta Rule over Sequence Length (https://arxiv.org/abs/2406.06484)*

在 DeltaNet 中, 其 hidden state 的更新方式为:
$$\begin{align*}
\mathrm{H}_t &= \mathrm{H}_{t-1} (I - \beta_t \boldsymbol{k}_t \boldsymbol{k}_t^\top) + \beta_v \boldsymbol{v}_t \boldsymbol{k}_t^\top \\
\end{align*}$$

该设计的初衷如下. 当我们想要去对 hidden state $\mathrm{H}_{t} := \mathrm{H}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\top$ 进行更新时, 事实上 $\mathrm{H}_{t-1}$ 的对应位置也是已经有一些历史的记忆了. 因此这里希望首先识别并清除掉历史的记忆  $\boldsymbol{v}_{t,\text{old}} := \mathrm{H}_{t-1} \boldsymbol{k}_t$, 再写入新的记忆:
$$\begin{aligned}
\mathrm{H}_t &:= \mathrm{H}_{t-1} - \beta_t \boldsymbol{v}_{t,\text{old}} \boldsymbol{k}_t^\top + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^\top\\
&= \mathrm{H}_{t-1} -\beta_t \mathrm{H}_{t-1} \boldsymbol{k}_t \boldsymbol{k}_t^\top + \beta_t \boldsymbol{v}_t \boldsymbol{k}_t^\top\\
&= \mathrm{H}_{t-1} - \beta_t (\mathrm{H}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t) \boldsymbol{k}_t^\top
\end{aligned}$$
这里 $\beta_t$ 是可学习的参数表示遗忘的程度. 

然而这里的最后一个式子:
$$
\mathrm{H}_{t} := \mathrm{H}_{t-1} - \beta_t (\mathrm{H}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t) \boldsymbol{k}_t^\top
$$
在某种意义上就是一个 Gradient Descent ! 其中 $\beta_t$ 就是学习率, 而 $(\mathrm{H}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^\top$ 是
$$
\mathcal L_t(\mathrm H) = \frac{1}{2} \|\mathrm{H} \boldsymbol{k}_t - \boldsymbol{v}_t\|^2
$$
之梯度. 而这个loss function的含义就是, 在我们更新 hidden state (即memory) 的时候, 我们希望通过 $\boldsymbol{k}_t$ 抽出的资讯和 $\boldsymbol{v}_t$ 尽量接近. 
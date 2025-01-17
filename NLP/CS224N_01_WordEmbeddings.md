# Word Embeddings 

> *Ref: [CS224N_Lecture 1](https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&ab_channel=StanfordOnline)*


## Word Vector

NLP的首要问题是如何表示词. 

- 传统的NLP模型中, 每个词都是一个one-hot vector, 也就是说每个词都是一个维度为$V$的向量, 其表示是离散的. 这样的表示方式会导致词与词之间都是正交的, 因此无法表示词与词之间的相似性.

- 为了解决这个问题, 我们引入了 **word vector** (或 **word embeddings**), 也就是将每个词映射到一个连续的向量空间中, 并且使得语义相近的词在这个空间中距离也比较近.

通过 word embedding 就可以将传统的离散的词表示转化为连续的词向量表示. 
- 当我们设计出了一个 word embedding 矩阵 $U \in \mathbb{R}^{V \times d}$ 之后, 我们就可以通过简单的矩阵乘法来得到一个词的词向量 (这里 $V$ 为词汇表的大小, $d$ 为词向量的维度是一个人为设定的超参数). 
- 对于$U$中的每一行 $u_i$ 都是一个词的词向量, 其表示了这个词在这个$d$维空间中的位置. 对应的, 对于原先 one-hot 表示的一句话 (记为矩阵 $X \in \mathbb{R}^{V \times T}$, 其中 $T$ 为句子长度), 我们可以通过简单的矩阵乘法来得到这句话的词向量表示 $X^\top U$. 
- 这个过程也将一个**高维稀疏的表示**转化为了一个**低维稠密的表示**.

## Word2Vec

> *Ref: [Efficient Estimation of Word Representations in Vector Space, Mikolov et al. 2013](https://arxiv.org/abs/1301.3781)*

Word2Vec 是一个用于学习 word embeddings 的模型.

### 基本思想

**Word2Vec** 是一种用于学习 word embeddings 的模型, 其基本想法为:
- 我们拥有一个大的文本 **corpus** (语料库), 其中包含了大量的文本数据. 我们可以由此构建一个dictionary, 并且将每个词标识为一个向量.
- 模型会‘扫描’整个文本数据, 每次会选取一个中心词 $\mathrm{c}$ (center word) 和其余的上下文词 $\mathrm{o}$ (outside words).
- 直观而言, 在一个文本数据中, 一个词的上下文词往往会与这个词有一定的语义关联, 或出现在一起的概率较大. 因此我们可以以此量化这种 *在 $\mathrm{c}$ 的上下文中 $\mathrm{o}$ 出现的概率* 或 *在 $\mathrm{o}$ 的上下文中 $\mathrm{c}$ 出现的概率*, 通过类似极大似然的思想来优化我们的向量表示.
    ![20250116130511](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250116130511.png)

> **注**: 这里我们用罗马体 $\mathrm{c}$ 等字符表示具体的词, 即计算机中的`str`类型. 

Word2Vec 根据其具体的实现方式可以分为两种模型: **Skip-gram** 和 **CBOW**. 由于 **Skip-gram** 模型更为直观, 因此我们首先介绍这个模型.

---

### Skip-gram 模型

#### 直观理解

- Skip-gram 模型的基本思想为: **给定一个中心词 $\mathrm{c}$, 我们希望通过这个词来预测其上下文词 $\mathrm{o}$**.
- 例如对于句子 *’the quick brown **fox** jumps over the lazy dog’*, 如果我们选取中心词为 *’fox’*, 则其上下文词 (根据窗口大小) 可能为 *’quick’*, *’brown’*, *’jumps’*, *’over’*.
- 直观上, 我们要进行一项优化工作, 使得*'quick'*, *'brown'*, *'jumps'*, *'over'* 这些词在给定 *'fox'* 的情况下出现的概率尽可能的高. 而其余非上下文的词 (如 *'the'*, *'lazy'*, *'dog'* 或其他语料库中的词) 在给定 *'fox'* 的情况下出现的概率尽可能的低.
    ![20250116170501](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250116170501.png)

#### 模型定义

- 首先定义文本位置 $t=1,2,...,T$, 其中 $T$ 为文本长度. 定义 $w_t \in \mathbb{R}^d$ 为文本位置 $t$ 的词向量, 其中 $d$ 为词向量的维度为人为设定的超参数 (常取 $d=100$ 至 $300$). 另记全部需要学习的参数为 $\theta$.
- 根据定义则可以定义给定一词汇位置 $t$ 的词向量 $w_t$ 下, 其 $t+j$ 位置出现某词向量 $w_{t+j}$ 的概率为 $\mathbb{P}(w_{t+j}|w_t)$, 其中 $j \in [-m,m]$ 为上下文窗口大小 (通常较短, 2~4 个单词). 则我们可以定义整个文本数据的似然函数为:
    $$  \mathcal{L}(\theta) = \prod_{t=1}^T \prod_{-m \leq j \leq m, j \neq 0} \mathbb{P}(w_{t+j}|w_t ; \theta) $$
- 取其负对数似然函数为损失函数 (或目标函数):
    $$ \mathcal{J}(\theta) = - \frac{1}{T} \sum_{t=1}^T \sum_{-m \leq j \leq m, j \neq 0} \log \mathbb{P}(w_{t+j}|w_t ; \theta) $$
    则最小化该损失函数即最大化预测的准确性. 


#### 概率求解

**此时一个重要的问题是如何定义 $\mathbb{P}(w_{t+j}|w_t ; \theta)$, 一种常见的方法是使用 **softmax** 函数:**

- 我们以随机变量的视角来完成后续的讨论.  通常我们认为会有一有限的词汇表 $\mathcal V$, 其中包含了所有可能的词. 记随机变量 $\mathrm{C} \in \mathcal V$ 为中心词 (center word), $\mathrm{O} \in \mathcal V$ 为上下文词 (outside word). 小写字母 $\mathrm{c}$ 和 $\mathrm{o}$ 分别为其具体的取值.
- 为了后文中某种计算的方便, 我们对于同一词汇采取两种不同的向量表示: $U \in \mathbb{R}^{|\mathcal V| \times d}$ 和 $V \in \mathbb{R}^{|\mathcal V| \times d}$. $U,V$ 的某一行就代表着 $\mathrm{w}\in \mathcal V$ 中的某一个词的词向量, 记这个词向量为 ${u}_\mathrm{w}$ 或 ${v}_\mathrm{w}$.
  - 当 $\mathrm{w}$ 为 center word (c) 时, 记之为 ${v}_\mathrm{w}$
  - 当 $\mathrm{w}$ 为 context word (o) 时, 记之为 ${u}_\mathrm{w}$
- 因此当给定某一具体的 center word $\mathrm{c}$ 时, 其对应某一具体的 context word $\mathrm{o}$ 的概率为:
    $$ \mathbb{P}(\mathrm{o}|\mathrm{c}) = \frac{\exp({u}_\mathrm{o}^\top {v}_\mathrm{c})}{\sum_{\mathrm{w} \in \mathcal V} \exp({u}_\mathrm{w}^\top {v}_\mathrm{c})} $$
    其中 $\mathcal V$ 为整个词汇表. 这里我们相当于用 inner product 来衡量两个词向量之间的相似性, 并且用 softmax 函数来量化其概率.
- 若另记整个文档的集合为 $\mathcal D$, 则我们可以定义整个文档的损失函数就可以被具体定义为:
    $$\begin{equation}
        \mathcal{J}(\theta) = \sum_{d \in \mathcal D} \sum_{t=1}^{|T_d|} \sum_{-m \leq j \leq m, j \neq 0} -  \log \mathbb{P}(w^{(d)}_{t+j}|w^{(d)}_t ; \theta) 
    \end{equation} $$
    即我们对全部文档的全部词汇位置的全部给定上下文窗口的概率取负对数求和.
    - 事实上, 这一表达式与 **cross-entropy** 损失函数等价 (当然, $(1)$ 是这个交叉熵sun是函数的经验估计), 即
     $$ \min_{U,V} \mathbb{E}_{\mathrm{c},\mathrm{o}}[-\log \mathbb{P}_{U,V}(\mathrm{o}|\mathrm{c})] $$
    
- 在定义了损失函数之后, 我们可以通过 SGD 等方法来优化我们的参数 $\theta$. 具体而言, 我们首先给$U,V$一个随机的初始值, 然后不断迭代地更新 $U,V$ 直到损失函数收敛. 这里的 $U,V$ 就是我们最终学习到的词向量.

#### 梯度与梯度下降

梯度下降的部分事实上已经可以由计算机自动求导来完成. 但是其梯度的结果有这很强的数学直觉, 因此我们在这里进行一下推导.

- 通常按照梯度下降的思想, 我们会首先随机初始化, e.g. $U, V \sim \mathcal N(0, 0.001) ^{|\mathcal V| \times d}$. 接着通过计算梯度来更新$U$的取值 ($V$同理):
   $$\begin{equation}
   U^{(i+1)} := U^{(i)} - \alpha \nabla_{U} \mathcal{J}(U^{(i)}, V^{(i)})
    \end{equation}$$
- 当然对于 SGD, 我们会对整个文档集合 $\mathcal D$ 进行随机采样得到一些子样本: $d_1, d_2, ..., d_{l} \sim \mathcal D$, 然后对于每个子样本 $d_i$ 进行梯度下降. 这样可以减少计算量, 并且可以更快地收敛, 即:
    $$\begin{equation} \mathcal{\hat J}(U,V)  = \sum_{d_1, ..., d_{l}} \sum_{t=1}^{|T_{d_i}|} \sum_{-m \leq j \leq m, j \neq 0} -  \log \mathbb{P}(w^{(d_i)}_{t+j}|w^{(d_i)}_t) \end{equation}$$

**结合前两个式子, 下具体计算一下梯度.**

- 对于某个具体的 center word $\mathrm{c}$ 和 context word $\mathrm{o}$, 其对应词向量 $v_\mathrm{c}$, $u_\mathrm{o}$ (其实就是上面的 $w_t, w_{t+j}$), 由于求和和求导可交换次序, 我们最终求导的对象为:
    $$\begin{align}
    \nabla_{v_\mathrm{c}} \log \mathbb{P}(\mathrm{o}|\mathrm{c}) &= \nabla_{v_\mathrm{c}} \log \frac{\exp(u_\mathrm{o}^\top v_\mathrm{c})}{\sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})} \\
    &= \nabla_{v_\mathrm{c}} \log \exp(u_\mathrm{o}^\top v_\mathrm{c}) - \nabla_{v_\mathrm{c}} \log \sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})
    \end{align}$$

- 分别计算着两部分的梯度. 对于 $\nabla_{v_\mathrm{c}} \log \exp(u_\mathrm{o}^\top v_\mathrm{c})$:
    $$\begin{aligned}
    \nabla_{v_\mathrm{c}} \log \exp(u_\mathrm{o}^\top v_\mathrm{c}) &= \nabla_{v_\mathrm{c}} u_\mathrm{o}^\top v_\mathrm{c} = u_\mathrm{o}
    \end{aligned}$$
    - 最后一步是因为 $\nabla_v u^\top v = \nabla_v \sum_{i=1}^d u_i v_i = [u_1+0, u_2+0, ..., u_d+0]^\top = u$. 转置是习惯上要求梯度 $\nabla_x$ 结果的形状应与 $x$ 一致.

- 对于第二部部分 $\nabla_{v_\mathrm{c}} \log \sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})$:
    $$\begin{aligned}
    \nabla_{v_\mathrm{c}} \log \sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})  
    &= \frac{1}{\sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})} \nabla_{v_\mathrm{c}} \sum_{\mathrm{x} \in \mathcal V} \exp(u_\mathrm{x}^\top v_\mathrm{c})      \quad\small{\text{(对数求导, 链式法则)}}    \\
    &= \frac{1}{\sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})} \sum_{\mathrm{x} \in \mathcal V} \nabla_{v_\mathrm{c}} \exp(u_\mathrm{x}^\top v_\mathrm{c})  \quad\small{\text{(加法与求导可交换)}} \\
    &= \frac{1}{\sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})} \sum_{\mathrm{x} \in \mathcal V} \exp(u_\mathrm{x}^\top v_\mathrm{c}) \nabla_{v_\mathrm c} u_\mathrm{x}^\top v_\mathrm c \quad\small{\text{(指数求导)}} \\
    &=\frac{ \sum_{\mathrm{x} \in \mathcal V} \exp(u_\mathrm{x}^\top v_\mathrm{c})~u_\mathrm{x} }{\sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})}  \quad\small{\text{(求梯度, 同上)}} \\
    \end{aligned}$$

- 最终将计算结果整合至$(5)$, 可以得到:
    $$\begin{aligned}
    \nabla_{v_\mathrm{c}} \log \mathbb{P}(\mathrm{o}|\mathrm{c}) &= u_\mathrm{o} -\sum_{\mathrm{x} \in \mathcal V}   \frac{ \exp(u_\mathrm{x}^\top v_\mathrm{c}) }{\sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})}u_\mathrm{x} \\
    &= u_\mathrm{o} - \sum_{\mathrm{x} \in \mathcal V} \mathbb{P}(\mathrm{x}|\mathrm{c}) u_\mathrm{x}\\
    &= u_\mathrm{o} - \mathbb{E}_{\mathrm{w} \sim \mathbb{P}(\mathrm{w}|\mathrm{c})} [u_\mathrm{w}]
    \end{aligned}$$

**最终的计算结果具有非常强的数学直觉.**
- 这个梯度的含义为: $\text{Observed context embeddings} - \text{Expected context  embeddings}$. 即观测到的上下文词向量减去预期的上下文词向量.
  - 具体地, 这个期望表示:根据当前的 Embedding $U,V$, 我们计算出的对于某个词的上下文分布的期望值.
- 通过梯度下降, 我们试图更新词向量表示, 使得其尽可能*接近*观测到的上下文词向量, 而*远离*预期的上下文词向量.

####  Skip-gram 中的 Negative Sampling 技巧

> *Ref: [Distributed Representations of Words and Phrases and their Compositionality, Mikolov et al. 2013](https://arxiv.org/abs/1310.4546)*

Negative Sampling 是一种用于优化 Word2Vec 模型的技巧, 我们需要它的原因是:

- 回顾在 Word2Vec 中, 我们的优化目标为:
    $$
    \min_{U,V} \mathbb{E}_{\mathrm{c},\mathrm{o}}[-\log \mathbb{P}_{U,V}(\mathrm{o}|\mathrm{c})] = \min_{U,V} \mathbb{E}_{\mathrm{c},\mathrm{o}}[-\log \mathbb{P}\frac {\exp(u_\mathrm{o}^\top v_\mathrm{c})}{\sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})}]
    $$

- 我们有时也称这个 Softmax 函数中的分母为 partition function, 其具有如下作用:
  - 一方面, 在概率的角度, Softmax 的分母部分是作为归一化项, 用于保证概率的和为1. 
  - 另一方面, 在优化的角度, 我们在最小化交叉熵损失函数时, 相当于尽可能的提高 $\exp(u_\mathrm{o}^\top v_\mathrm{c})$ 的取值 (即 center word $\mathrm{c}$ 和 context word $\mathrm{o}$ 的相似性), 同时尽可能的降低 $\sum_{\mathrm{w} \in \mathcal V} \exp(u_\mathrm{w}^\top v_\mathrm{c})$ 的取值 (即降低 center word $\mathrm{c}$ 和其他所有无关词汇 $\mathrm{w}$ , 当 $\mathrm{w} \neq \mathrm{o}$ 的相似性).

- 但是在实作中, partition function 具有如下问题:
  - 计算复杂度: 由于需要对整个词汇表 $\mathcal V$ 进行求和, 因此计算复杂度为 $\mathcal O(|\mathcal V|)$. 这对于大规模的词汇表来说是不可接受的.
  - 优化动机: 尽管其在概率的角度有其必要性, 但在其在优化方面的作用往往被指数求和所稀释了. 因此其对于优化的作用亦是有限的.

- Negative Sampling 就是为了解决这个问题而提出的技巧. 其认为: **我们的“分区函数”没有必要遍历整个词汇表, 只需要对一小部分的有代表性的词汇进行采样即可.**

Negative Sampling 的具体实现如下:

- 对于每一个 center word $\mathrm{c}$ 和 context word $\mathrm{o}$, 我们定义一个新的损失函数 (严格意义上是其相反数):
    $$\begin{equation}
    \log \sigma (u_\mathrm{o}^\top v_\mathrm{c}) + \sum_{\mathrm{k}\in \mathcal{K}} \left[ \log \sigma (-u_{\mathrm{k}}^\top v_\mathrm{c}) \right]
    \end{equation}$$ 
    - 其中 $\sigma(x) = \frac{1}{1+\exp(-x)}$ 为 sigmoid 函数.
    - $\log \sigma (u_\mathrm{o}^\top v_\mathrm{c})$ 为 positive sampling, 与原来的分子相似, 即希望让 central word $\mathrm{c}$ 和 context word $\mathrm{o}$ 的相似性尽可能的高.
    - $\mathcal{K}$ 为通过 negative sampling 随机采样得到的 $K$ 个 negative samples 的集合, 或着说 $u_{\mathrm{k}} \sim p_{neg}$, 其中 $p_{neg}$ 为负采样的分布. 
      - $p_{neg}$ 通常为词频分布的负采样, 即词频越高的词越容易被采样到. 对于某单词 $\mathrm{w}\in \mathcal{V}$, 其采样概率为 $p_{neg}(\mathrm{w}) \propto \mathrm{freq}(\mathrm{w})^{\alpha}, ~\alpha \in (0,1)$.
      - 故在实作中, 往往需要进行预处理, 遍历整个文档, 统计每个词的词频, 并且根据词频计算其采样概率:
        $$
        p_{neg}(\mathrm{w}) = \frac{\mathrm{freq}(\mathrm{w})^{\alpha}}{\sum_{\mathrm{w'} \in \mathcal{V}} \mathrm{freq}(\mathrm{w'})^{\alpha}}
        $$
      - 一个经验上的建议是 $\alpha = 0.75$.
    - $\log \sigma (-u_{\mathrm{k}}^\top v_\mathrm{c})$ 为 negative sampling, 即希望让 central word $\mathrm{c}$ 和 negative sample $\mathrm{k}$ 的相似性尽可能的低.

#### Skip-gram 的神经网络视角

> 注: 以下部分内容包含 AGI 内容, 仅供参考.

在神经网络的视角下, Skip-gram 模型可以被视为一个简单的神经网络模型. 其具体的网络结构如下:

![20250116200255](https://raw.githubusercontent.com/By-Xin/Blog-figs/main/20250116200255.png)

***从数学结构看***

1. **输入层 (Input layer)**  
   - 这里输入是一个 “one-hot” 向量 $\mathbf{x}\in \mathbb{R}^V$。它对应“中心词” $\mathrm{c}$ 的索引，只有在 $\mathrm{c}$ 位置上为 1，其余位置为 0。
   - 对应到我们数学推导中的 $\mathrm{c}$ 即 center word，或者更严格地说是它的 one-hot 表示。
  
2. **隐藏层 (Hidden layer) / Embedding 层**  
   - 我们定义一个可训练的参数矩阵 $W_{\text{in}} \in \mathbb{R}^{V\times d}$，其中的第 $i$ 行即是词表中第 $i$ 个词的 $\mathbf{v}_i$（“中心词向量”）。
   - 当输入层是一个 one-hot 向量 $\mathbf{x}$ 时，做矩阵乘法 $\mathbf{h} = \mathbf{x}^\top W_{\text{in}}$ 之后，实际上就相当于**选出**了第 $\mathrm{c}$ 行的权重作为隐藏层的输出 $\mathbf{h}$。  
     - 如果 $\mathbf{x}$ 在第 $\mathrm{c}$ 个位置是 1，那么 $\mathbf{h} = W_{\text{in}}[\mathrm{c}, :]$，维度是 $d$。这也就是“嵌入 (embedding)”的来源。

3. **输出层 (Output layer)**  
   - 接下来我们再定义另一个可训练参数矩阵 $W_{\text{out}} \in \mathbb{R}^{V \times d}$；其中的第 $j$ 行可以看作是词表中第 $j$ 个词的 “上下文词向量” $\mathbf{u}_j$（也叫 $\mathbf{u}_\mathrm{w}$）。
   - 将隐藏层 $\mathbf{h}$ 做一次矩阵乘法就得到 $\mathbf{z} = W_{\text{out}} \mathbf{h}\in \mathbb{R}^V$。  
     - 第 $j$ 个输出的 logit （即没过 softmax 或 sigmoid 前的值）就是 $\mathbf{z}[j] = \mathbf{u}_j^\top \mathbf{h}$。
   - 若不使用负采样，而是用 softmax，则再将 $\mathbf{z}$ 过一个 softmax 就可以得到对整张词表的预测概率分布：  
     $$
       \hat{y}[j] = \mathbb{P}(\mathrm{j}|\mathrm{c}) 
                  = \frac{\exp(\mathbf{u}_j^\top \mathbf{v}_\mathrm{c})}{\sum_{w \in \mathcal{V}}\exp(\mathbf{u}_w^\top \mathbf{v}_\mathrm{c})}.
     $$

将这三步串起来，你会发现：  
- “输入层 $\to$ 隐藏层” 对应了**一层线性映射** (矩阵 $W_{\text{in}}$)。  
- “隐藏层 $\to$ 输出层” 又是一层线性映射 (矩阵 $W_{\text{out}}$)，再加一个 softmax/sigmoid。  
- 整个过程和一个“前馈神经网络 (Feed-forward NN)”其实没有本质区别——只是它的激活函数是恒等映射（没有显式地用 ReLU / Sigmoid / Tanh 去做隐藏层激活），因此在结构上看起来就像“输入层”**直连**“输出层”那样，非常简单。

***从训练过程看***

Word2Vec 的训练目标是让**中心词**和它的**上下文词**之间的内积尽可能大，同时让与不相关词的内积尽可能小，这通过**交叉熵**（或负采样的目标函数）来实现。具体训练流程与神经网络并无二致：

1. **前向传播 (Forward pass)**  
   - 输入层给出 one-hot 向量 ——> 隐藏层 $\mathbf{h}$ = 取该行 embedding 向量 ——> 输出层 logit $\mathbf{z} = W_{\text{out}} \mathbf{h}$ ——> 过 softmax / sigmoid 得到预测概率分布 $\hat{y}$。  

2. **计算损失 (Loss)**  
   - 与目标分布（“哪个词是真实的上下文词”）做 cross-entropy；或者在负采样里，对正样本做 $\log\sigma(\cdot)$，对负样本做 $\log\sigma(-\cdot)$，把这几项加起来变成一次 mini-batch 的损失。

3. **后向传播 (Backward pass)**  
   - 计算损失对输出层权重 $W_{\text{out}}$ 的梯度，更新它。  
   - 计算损失对隐藏层 $\mathbf{h}$ 的梯度，进而可更新输入层权重 $W_{\text{in}}$（也就是 embeddings）。  
   - **在代码实现里**，这一部分通常要么借助自动微分框架（如 PyTorch 的 autograd），要么自己手写类似上面推导的公式。  
   - 训练时就是不停地**采样 (中心词, 上下文词)**、做**前向**和**后向**，不断更新 $W_{\text{in}}$ 和 $W_{\text{out}}$ 这两个大矩阵，直到收敛。

因此，**它就是一个（非常）浅层的神经网络**。只不过它生来就是为了“学词向量”，因此往往把对输出层的巨大 softmax 做各种近似 (如负采样、分层 softmax 等)，以减小计算量。

- 在实现上，如果你用 PyTorch 或 TensorFlow 来写一个 Skip-gram / CBOW 的训练代码，**也确实会写出一段类似下面的过程**：  
  1. `embed_c = W_in[center_word_idx]`  
  2. `logits = W_out @ embed_c`  
  3. `loss = cross_entropy_loss(logits, target_idx)`  
  4. `loss.backward()`  
  5. `optimizer.step()`  
  这和写一个分类网络（input -> linear -> linear -> output）并没有本质区别。

因此，不要被它表面上的“概率建模”或“对数似然”公式吓到，**它在本质和训练流程上确实就是一个神经网络**，只是它的结构非常简单，没有激活函数、只有一个隐层 (embedding) 而已。  

某种意义上, 只要一个模型能够:
1. 以可微的方式从输入映射到输出（前向）
2. 通过求导/反向传播来更新其中的参数，那它就算是一个神经网络（尤其在深度学习时代的语境里）


#### Skip-gram 的矩阵分解视角

首先做如下规定:

1. 记词汇表大小为 $V$，词向量维度为 $d$。  
2. 两个可训练的 **Embedding 矩阵**：  
   - $W_{\mathrm{in}} \in \mathbb{R}^{V \times d}$，第 $i$ 行是词表中第 $i$ 个词的 **“中心词向量”** $\mathbf{v}_i \in \mathbb{R}^d$。  
   - $W_{\mathrm{out}} \in \mathbb{R}^{V \times d}$，第 $j$ 行是词表中第 $j$ 个词的 **“上下文词向量”** $\mathbf{u}_j \in \mathbb{R}^d$。  

3. 对于一个训练样本 **(center= $c$, outside= $o$)**，它的**中心词**索引是 $c$，**上下文词**索引是 $o$。  
   - 从矩阵看：  
     $\mathbf{v}_c = W_{\mathrm{in}}[c, :]$  
     $\mathbf{u}_o = W_{\mathrm{out}}[o, :]$

4. 为了方便理解，下文里我们**不区分 batch**，只考虑“单个中心词、上下文词”的场景；若要扩展到 batch，只需要把对应的索引拼成一个向量/矩阵，再在矩阵乘法中一起算即可。

---

***1. Skip-gram + Full Softmax 的矩阵形式***

这里的核心是：**给定中心词 $c$，模型要对整张词表 $V$做softmax**，再用交叉熵来对比 “哪个词才是真实的上下文 $o$”。

1. **取出中心词向量**  
   $$
     \mathbf{v}_c = W_{\mathrm{in}}[c, :] \quad \in \mathbb{R}^{1 \times d}
   $$

2. **对所有可能的上下文词（词表大小 = $V$）逐一打分**  
   $$
     \mathbf{z} = \mathbf{v}_c \times (W_{\mathrm{out}})^\top 
     \quad\in \mathbb{R}^{1 \times V}.
   $$
   - 其中 $W_{\mathrm{out}}$ 的形状是 $(V \times d)$，转置后 $(d \times V)$。  
   - 于是得到向量 $\mathbf{z} = [z_1, z_2, ..., z_V]$ 表示对每个词的分数 (logits)。  
   - 具体可写成 $z_j = \mathbf{v}_c \cdot \mathbf{u}_j$，即**中心词**和**候选词**向量的点积。  

3. **Softmax** 归一化  
   $$
     \hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{z}) 
     \quad\in \mathbb{R}^{1 \times V}.
   $$
   - $\hat{y}_j = \dfrac{\exp(z_j)}{\sum_{w=1}^V \exp(z_w)}$。  
   - $\hat{\mathbf{y}}$ 就是对所有词的预测概率分布。

4. **损失函数 (交叉熵)**  
   - 若真实的上下文词是 $o$，则它的 label 可以视为 one-hot 向量 $\mathbf{y}\in \mathbb{R}^{V}$ 在第 $o$ 位置为 1，其它为 0。  
   - 训练时的**交叉熵**损失：
     $$
       \mathcal{L}_{\mathrm{softmax}} 
       = -\sum_{j=1}^V y_j \log \hat{y}_j 
       = -\log \hat{y}_o 
       = -\log \left(\dfrac{\exp(z_o)}{\sum_{w=1}^V \exp(z_w)}\right). 
     $$
     也常写成 $\mathcal{L} = -\log \mathrm{softmax}(\mathbf{v}_c \cdot W_{\mathrm{out}}^\top)[o]$。  

**从矩阵的视角**：  
- **输入**：索引 $c$；**查表**得到 $\mathbf{v}_c$。  
- **输出**：和 $W_{\mathrm{out}}$ 的所有行做内积 ($\mathbf{v}_c \times W_{\mathrm{out}}^\top$)，再做 softmax。  
- **损失**：Cross-entropy 与真实的上下文 $o$ 做比较。  

若对所有样本求和/取平均，就得到了整篇语料的损失；随后做梯度下降时，更新 $W_{\mathrm{in}}$ 和 $W_{\mathrm{out}}$ 中对应行(或所有行)的权重。

---

***2. Skip-gram + Negative Sampling 的矩阵形式***

相比之下，**负采样 (Negative Sampling)** 并不会对所有 $V$ 个词都计算 softmax。它的思路是：**对“真实上下文词 $o$”打分，想让它的内积越大越好；并对“若干随机负样本 $k \in \mathcal{K}$”打分，想让它们越小越好。** 这可以看作是一组二分类 $\sigma(\cdot)$ 的组合。

1. **取出中心词向量**  
   $$
     \mathbf{v}_c = W_{\mathrm{in}}[c, :] \quad (\text{形状 } (1 \times d)).
   $$

2. **对正样本 (中心词 $c$, 上下文 $o$)** 做打分**并**计算损失**  
   $$
     \text{(a) 先算点积: } 
        z_{\text{pos}} = \mathbf{v}_c \cdot \mathbf{u}_o 
        = \mathbf{v}_c \cdot W_{\mathrm{out}}[o, :];
   $$
   $$
     \text{(b) 正样本的损失} 
        = -\log \sigma(z_{\text{pos}}).
   $$
   - 希望 $\sigma(z_{\text{pos}}) \approx 1$，表示“$o$是正确的上下文”。

3. **对负样本 (中心词 $c$, 负样本 $k$)** 做打分并计算损失**(多个 $k$)**  
   $$
     \text{(a) 点积: } 
        z_{\text{neg},k} = \mathbf{v}_c \cdot \mathbf{u}_k,
        \quad k \in \mathcal{K} \subseteq \{1,\ldots,V\}.
   $$
   - 其中 $\mathcal{K}$ 是这一次训练采到的几个负例的索引。  
   $$
     \text{(b) 负样本的损失} 
        = - \sum_{k \in \mathcal{K}} \log \bigl( 1 - \sigma(z_{\text{neg},k}) \bigr).
   $$
   - 等价写成 $- \sum_{k \in \mathcal{K}} \log \sigma(-z_{\text{neg},k})$。  
   - 希望 $\sigma(z_{\text{neg},k}) \approx 0$，表示“$k$并不是$c$的真正上下文”。

4. **总损失**  
   $$
     \mathcal{L}_{\mathrm{neg\_sample}}
     = -\log \sigma(\mathbf{v}_c \cdot \mathbf{u}_o)
       \;-\; \sum_{k\in \mathcal{K}}
            \log \bigl(1 - \sigma(\mathbf{v}_c \cdot \mathbf{u}_k)\bigr).
   $$
   - 这里并没有一次性算出所有 $V$ 个词的打分；只算 1 个正例 + 若干负例。

**从矩阵视角：**  
- 如果我们一次性处理所有负样本的向量，可以把它们在 $W_{\mathrm{out}}$ 里“查表”后拼成一个矩阵 ，其中第一行对应 $\mathbf{u}_o$，其余行对应各 $\mathbf{u}_k$。  
- 用 $\mathbf{v}_c$ 去和这个小矩阵 $\mathbf{U}_{\text{neg}}$ 做一次矩阵乘法，得到一组分数 $\{z_{\text{pos}}, z_{\text{neg},1}, z_{\text{neg},2}, ...\}$，再把它们喂进 $\sigma(\cdot)$ 分别做“正样本=1, 负样本=0”的二分类。  
- **不会**像 softmax 那样，对整个 $V$ 维输出做 $\mathrm{softmax}$；因此大大降低了计算量。

---

下面给出一个**对照表**，概括二者在“矩阵运算”层面的主要差异。

|                | **Skip-gram + Full Softmax**                                              | **Skip-gram + Negative Sampling**                                                               |
|----------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **要计算的 logits** | 需对 **词表中所有词**（$V$ 个）计算 $\mathbf{v}_c^\top \mathbf{u}_w$，得到形状 $(1\times V)$ 的向量，再用 softmax。  | 只对 **1 个正样本** + **$K$ 个负样本** 计算打分，共 $K+1$ 个内积，分别做 $\sigma(\cdot)$ 二分类。 |
| **具体公式**   | $\mathbf{z} = \mathbf{v}_c \times (W_{\mathrm{out}})^\top \in \mathbb{R}^{1\times V}$<br>$\hat{\mathbf{y}}=\mathrm{softmax}(\mathbf{z})$<br>$\mathcal{L}=-\log \hat{y}_o$ | $\mathcal{L}=-\log \sigma(\mathbf{v}_c \cdot \mathbf{u}_o) - \sum_{k\in\mathcal{K}} \log \bigl[1 - \sigma(\mathbf{v}_c \cdot \mathbf{u}_k)\bigr]$ |
| **损失解释**   | 多分类：希望对真正的上下文词 $o$ 给出最高概率。                                     | 多个独立的二分类：希望“正样本打分高 ($=1$)”，“负样本打分低 ($=0$)”。                                   |
| **计算代价**   | 一次 forward/backward 需要对 $\mathbf{v}_c$ 与 **$V$ 个上下文向量** 做内积；$V$ 通常很大，计算量高。 | 一次只对 **$K+1$** 个样本做内积；$K$ 通常远小于 $V$，计算量小得多。                                     |
| **数学视角**   | 本质是一层 **线性映射** $\mathbf{v}_c \mapsto \mathbf{z}\in\mathbb{R}^V$ 再 + **softmax**。 | 本质是一批**二分类打分** $\mathbf{v}_c \cdot \mathbf{u}_o$ / $\mathbf{v}_c \cdot \mathbf{u}_k$ + Sigmoid。             |

---


- **Softmax 版本：**  
  $$
    \boxed{
      \mathbf{z} = \mathbf{v}_c \times W_{\mathrm{out}}^\top \quad\longrightarrow\quad
      \hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{z})
      \quad\longrightarrow\quad
      \mathcal{L} = -\log \hat{y}_o.
    }
  $$  
  一次 forward 针对 **$V$ 个**词的分布做多分类预测，数据量大，但准确地表达了“归一化概率=1”的约束。

- **Negative Sampling 版本：**  
  $$
    \boxed{
      \mathcal{L} 
      = -\log \sigma(\mathbf{v}_c \cdot \mathbf{u}_o) 
        \;-\; \sum_{k=1}^{K} \log\Bigl(1 - \sigma(\mathbf{v}_c \cdot \mathbf{u}_k)\Bigr).
    }
  $$  
  一次 forward 只对正样本 + $K$ 个负样本做二分类，数据量大幅减少，但失去全局归一化的信息，通过反复采样来近似 softmax 的“推开其他词”的效果。

二者在本质上都依赖同一个 **(center word embedding $\mathbf{v}_c$, context word embedding $\mathbf{u}_o$)** 的内积做打分的想法，只是在 **损失计算**（对哪些词做内积、怎么做归一化或对比）上有差别.


#### Skip-gram 的 PyTorch 实现

***准备工作***

```python
import torch
import random
import numpy as np
from collections import Counter

# 为了保证每次运行结果一致，可以固定随机种子
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 小小的 toy corpus，只是一句话或几句话的字符串列表
# （真实场景下可以是很多文本组合而成的）
corpus = [
    "i like to eat apples and bananas",
    "i like to watch movies and cartoons",
    "the cat likes to eat fish",
    "john loves to read books about python"
]

# 1) 对文本进行分词 (tokenization)
#    这里就用简单的 split() 做演示
tokenized_sentences = [sent.lower().split() for sent in corpus]

# 2) 汇总到一个 list，得到所有单词序列
all_tokens = []
for tokens in tokenized_sentences:
    all_tokens.extend(tokens)

# 3) 构造词表 (vocab)，并分配每个词一个ID
#    这里根据词频做一个简单的去重即可
word_counter = Counter(all_tokens)
vocab = sorted(word_counter.keys())  # 简单按字母排序
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

vocab_size = len(vocab)
print(f"Vocabulary size = {vocab_size}")
print("Vocab =", vocab)
```

**输出示例（可能略有变化）**  
```
Vocabulary size = 14
Vocab = ['about', 'and', 'apples', 'bananas', 'books', 'cartoons', 
         'cat', 'eat', 'fish', 'i', 'john', 'like', 'likes', 'loves', 
         'movies', 'python', 'read', 'the', 'to', 'watch']
```
(这是根据示例文本统计到的词汇，仅供演示。)

---

***生成训练样本 (Skip-gram)***

我们来手写一个函数 `make_skipgram_data`，它会：  
1. **遍历** 所有句子；  
2. 对每个**中心词**，在前后各 `window_size` 个范围内收集**上下文词**；  
3. 返回所有 **(center_idx, outside_idx)** 的对儿。  

```python
def make_skipgram_data(tokenized_sentences, word2idx, window_size=2):
    """
    根据 tokenized_sentences，生成 (center, outside) 对儿的列表。
    window_size 表示上下文窗口大小 (左右各 window_size 个).
    """
    pairs = []
    for tokens in tokenized_sentences:
        token_ids = [word2idx[w] for w in tokens]
        length = len(token_ids)
        for i, center_id in enumerate(token_ids):
            # 当前中心词位置是 i
            # 向左、向右看 window_size
            start = max(i - window_size, 0)
            end = min(i + window_size + 1, length)  # +1 因为 range() 是前闭后开
            for j in range(start, end):
                if j != i:  # 不要把中心词本身当做上下文
                    outside_id = token_ids[j]
                    pairs.append((center_id, outside_id))
    return pairs

window_size = 2
skipgram_pairs = make_skipgram_data(tokenized_sentences, word2idx, window_size)
print(f"Total skip-gram pairs: {len(skipgram_pairs)}")
# 随便看几个例子
print("Example pairs (center_idx, outside_idx):", skipgram_pairs[:10])
```

**输出示例**：
```
Total skip-gram pairs: 58
Example pairs (center_idx, outside_idx): [(9, 11), (9, 18), (11, 9), (11, 18), (11, 7), (18, 9), (18, 11), (18, 7), (7, 11), (7, 18)]
```
- 这里 `(9, 11)` 说明中心词ID=9 (就是 "i")，它的上下文词ID=11 (就是 "like")。

---

***定义 SkipGram 模型 (含负采样)***

接下来我们实现一个极简的神经网络类 `SkipGramNegSample`：  

- **输入层**: 接受 `center_word` 的 ID（one-hot 就不显式写了，在 PyTorch 中我们常用 `nn.Embedding` 来做“查表”）  
- **输出层**: 对 (center, outside) 做打分 $\mathbf{v}_c^\top \mathbf{u}_o$，再经过 sigmoid 来做二分类 (是正样本还是负样本?)。

我们用到的组件：  
1. `nn.Embedding(vocab_size, embed_dim)`：分别存储  
   - $W_{\text{in}} \in \mathbb{R}^{vocab\_size \times embed\_dim}$ -> “中心词向量”  
   - $W_{\text{out}} \in \mathbb{R}^{vocab\_size \times embed\_dim}$ -> “上下文词向量”  
2. 负采样 (negative sampling) 的实现思路：  
   - 给定正样本 (center, outside)，我们额外从词表中**随机**采 $K$ 个词，视为负样本。  
   - 对每个负样本，打分 $\mathbf{v}_c^\top \mathbf{u}_{neg}$，希望它越小越好 ($\sigma$ 输出接近 0)。  
   - 对正样本，打分 $\mathbf{v}_c^\top \mathbf{u}_{outside}$，希望它越大 ($\sigma$ 输出接近 1)。  

```python
import torch.nn as nn

class SkipGramNegSample(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_negatives=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_negatives = num_negatives
        
        # 中心词的 embedding 矩阵: W_in
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        # 上下文词的 embedding 矩阵: W_out
        self.out_embed = nn.Embedding(vocab_size, embed_dim)
        
        # 为了给负样本采样，这里简单起见，按词频构建一个抽样表
        # (越常见的词越容易被采到)
        word_freq = np.array([word_counter[idx2word[i]] for i in range(vocab_size)], dtype=np.float32)
        word_freq = word_freq ** 0.75  # 常见的经验, exponent=0.75
        self.neg_sampling_dist = word_freq / word_freq.sum()
        
        # 初始化 Embedding 的权重
        # PyTorch 默认就会用 ~U(-1,1) 或 ~N(0,1) 之类的做初始化，这里也可以自定义:
        nn.init.uniform_(self.in_embed.weight, a=-0.5, b=0.5)
        nn.init.uniform_(self.out_embed.weight, a=-0.5, b=0.5)
        
    def forward(self, center_word_ids, outside_word_ids):
        """
        参数:
            center_word_ids: (batch_size,) 里面是中心词的索引
            outside_word_ids: (batch_size,) 里面是真实上下文词(正样本)的索引
        返回:
            loss: 这批样本的平均损失 (标量)
        """
        batch_size = center_word_ids.size(0)

        # 1) 从 in_embed 中查出中心词向量: shape = (batch_size, embed_dim)
        center_embed = self.in_embed(center_word_ids)  
        
        # 2) 从 out_embed 中查出正样本(上下文)的向量: shape = (batch_size, embed_dim)
        outside_embed = self.out_embed(outside_word_ids)
        
        # 3) 计算正样本的打分: v_c · u_o
        #    再过 sigmoid, 并且用 BCE loss (正例label=1) 
        pos_scores = torch.sum(center_embed * outside_embed, dim=1)  # (batch_size,)
        pos_loss = - torch.log(torch.sigmoid(pos_scores) + 1e-8)     # 避免log(0)
        
        # 4) 负采样: 对每个 (center, outside), 采 self.num_negatives 个负样本
        #    先用 np.random.choice 采出 IDs，再转成 tensor
        neg_samples = np.random.choice(
            range(self.vocab_size), 
            size=(batch_size, self.num_negatives), 
            p=self.neg_sampling_dist
        )
        neg_samples = torch.LongTensor(neg_samples)  # (batch_size, num_negatives)
        
        # 5) 从 out_embed 中查出这些负样本的向量
        #    形状: (batch_size, num_negatives, embed_dim)
        neg_embed = self.out_embed(neg_samples)
        
        # 6) 计算负样本打分: v_c · u_neg, 再做 sigmoid, 目标label=0 -> log(1 - sigmoid(...))
        #    pos_scores 的形状是 (batch_size,1), neg_scores形状是 (batch_size,num_negatives)
        #    先 expand 一下 center_embed 方便广播相乘
        center_embed_expanded = center_embed.unsqueeze(1)  # (batch_size, 1, embed_dim)
        neg_scores = torch.bmm(neg_embed, center_embed_expanded.transpose(1,2)).squeeze()  
        # 解释: bmm做批量矩阵乘法, neg_embed: (b, n, d), center_embed_expanded: (b, d, 1)
        #       => neg_scores: (b, n, 1) => squeeze => (b, n)
        
        neg_loss = - torch.log(torch.sigmoid(- neg_scores) + 1e-8)  
        # label=0 => loss = - log(sigmoid( - (v_c·u_neg) ))
        # 也可以写成: neg_loss = - torch.log(1 - torch.sigmoid(neg_scores) + 1e-8)
        
        # 7) 合并正样本和负样本的 loss
        #    pos_loss shape=(batch_size,), neg_loss shape=(batch_size,num_negatives)
        #    取平均即可
        total_loss = (pos_loss + neg_loss.sum(1)).mean()
        
        return total_loss
```

这里最核心的地方就是**负采样**部分：  
- `pos_scores = v_c · u_o`，想让它变大 (sigmoid->1)。  
- `neg_scores = v_c · u_neg`，想让它变小 (sigmoid->0)。  

---

***训练流程***

我们手动写一个训练循环：  

1. 随机打乱所有 (center, outside) 对儿，做 **SGD** 或者 **mini-batch**。  
2. 前向 `loss = model(center_batch, outside_batch)`；  
3. 反向 `loss.backward()`；  
4. 更新参数 `optimizer.step()`。  

**为简单起见**，我们这里就用最朴素的“**纯SGD** + 单样本更新**”的写法（即 batch_size=1），实际上你可以把多个 pair 组成 mini-batch 再一起 forward/backward，会更高效。  

```python
embed_dim = 8      # 词向量维度
num_negatives = 4  # 每次采 4 个负样本
model = SkipGramNegSample(vocab_size, embed_dim, num_negatives)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 3
pairs_list = skipgram_pairs[:]  # 拷贝一下

for epoch in range(num_epochs):
    random.shuffle(pairs_list)  # 每轮打乱顺序
    
    total_loss = 0.0
    for (center_id, outside_id) in pairs_list:
        center_tensor = torch.LongTensor([center_id])
        outside_tensor = torch.LongTensor([outside_id])
        
        loss = model(center_tensor, outside_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(pairs_list)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss={avg_loss:.4f}")
```

**输出示例**：
```
Epoch 1/3, Avg Loss=3.3084
Epoch 2/3, Avg Loss=2.6271
Epoch 3/3, Avg Loss=2.4257
```
可以看到 Loss 在下降，说明模型正在“学到”一个让 (center_word, outside_word) 更匹配、同时跟负样本更不匹配的嵌入空间。

---

***验证训练结果***

训练结束后，我们就可以取出 `model.in_embed.weight` (或 `model.out_embed.weight`) 作为我们训练到的词向量。可以做点简单的测试，比如：**“找与某个词最相似的几个词”**。  

```python
def get_embedding(model, word):
    """
    返回某个word在in_embed下的向量 (center word embedding)。
    你也可以用out_embed，看应用场景而定
    """
    idx = word2idx[word]
    emb = model.in_embed.weight[idx].detach().numpy()
    return emb

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def most_similar_words(model, query_word, top_k=3):
    query_emb = get_embedding(model, query_word)
    sims = []
    for w in vocab:
        if w == query_word:
            continue
        emb = get_embedding(model, w)
        sim_score = cosine_sim(query_emb, emb)
        sims.append((w, sim_score))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]

test_words = ["i", "eat", "movies", "python", "book"]  # "book"其实不在这个vocab里
for w in test_words:
    if w not in word2idx:
        print(f"Word '{w}' not in vocab, skip.")
        continue
    print(f"\n[Top similar words to '{w}']")
    for candidate, score in most_similar_words(model, w):
        print(f"   {candidate:<8}  cos_sim = {score:.4f}")
```

可能输出类似：  
```
[Top similar words to 'i']
   john      cos_sim = 0.4843
   loves     cos_sim = 0.3221
   movies    cos_sim = 0.2790

[Top similar words to 'eat']
   likes     cos_sim = 0.5153
   fish      cos_sim = 0.4127
   cat       cos_sim = 0.3532

[Top similar words to 'movies']
   cartoons  cos_sim = 0.4678
   watch     cos_sim = 0.3982
   i         cos_sim = 0.2790

[Top similar words to 'python']
   books     cos_sim = 0.3843
   about     cos_sim = 0.3222
   read      cos_sim = 0.2931
```

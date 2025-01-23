# Word Embeddings 2: Co-occurrence Matrix, PMI, GloVe

> *Ref: [CS224N_Lecture 2](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes02-wordvecs2.pdf)*

## Co-occurrence Matrix 共现矩阵

### Intuition

传统的词语表示有两种方法:
- 基于词频计数与矩阵分解的方法
  - 例如 LSA (Latent Semantic Analysis)
  - 优点: 利用语料库的全局信息
  - 缺点: 难以表示语义关系, 如 "$\text{king} - \text{man} + \text{woman} \approx \text{queen}$"
- 基于预测的浅层模型
  - 例如 word2vec
  - 优点: 可以表示语义关系和复杂的语言结构
  - 缺点: 无法利用全局信息

尤其是在 word2vec 中, 我们相当于是在一遍一遍地扫描语料库, 以预测词语的上下文. 而一种更为直觉的想法是: 我们在训练的开始就创建一个共现(共同出现)矩阵, 用来表示词语之间的共现关系. 这样一来, 我们就可以利用全局信息来学习词向量.

### Notations & Remarks

下给出 co-occurrence matrix 具体的符号及定义:
- $X$: co-occurrence matrix, $X_{ij}$ 表示以$i$为中心词, $j$在其上下文中出现的次数


在构造 co-occurrence matrix 时, 我们同样可以选择窗口大小 (或是基于全文) 来定义上下文. 一个通常的窗口大小约为 $5\sim10$. 同时注意到, 对于 co-occurrence matrix, 由于并不区分词语的顺序, 因此其是一个**对称矩阵**.

需要指出, 创立这样一个 co-occurrence matrix 需要完整扫描整个语料库. 对于一个大型语料库, 其计算和储存成本较高, 尽管这在整个训练周期中只需 进行一次. 并且在大型语料库中 (即在高维的情况下), co-occurrence matrix 往往也会变得非常稀疏, 这也会影响模型的 robustness. 因此对于 co-occurrence matrix, 其很重要的一个问题就是信息的压缩与降维, 将其转化为更为紧凑的表示. 

### Naive Solution for Dimensionality Reduction: SVD

SVD (Singular Value Decomposition) 是一种常用的矩阵分解方法. 该定理可叙述如下: 

***Theorem (SVD)***: 给定任意一个矩阵 $X\in\mathbb{R}^{m\times n}$, 可以被分解为:
$$
X = U \Sigma V^\top
$$
其中:
- $U\in\mathbb{R}^{m\times m}$ 是正交矩阵 (即 $U^\top U=I$), 其列向量为 $X X^\top$ 的特征向量.
- $V\in\mathbb{R}^{n\times n}$ 是正交矩阵, 其列向量为 $X^\top X$ 的特征向量.
- $\Sigma\in\mathbb{R}^{m\times n}$ 是对角矩阵, 对角线上的元素称为奇异值 (singular values), 且按顺序依次满足 $\sigma_1  \geq \ldots \geq \sigma_k \geq \cdots \sigma_p \ge 0$.
  - Singular values 的定义是 $\sigma_i = \sqrt{\lambda_i}$, 其中 $\lambda_i$ 是 $X^\top X$ 的特征值.

对于这样的一个 SVD 分解, 我们可以取前 $k$ 个较大的奇异值 (对应 $U, \Sigma, V$ 的前 $k$ 列, 记为 $U_k, \Sigma_k, V_k$), 从而得到 $X$ 的一个低秩近似:
$$
\hat X \approx U_k \Sigma_k V_k^\top
$$

这个分解的作用是, 将原先的 $X\in \mathbb{R}^{m\times n}$ 降维为 $\hat X\in \mathbb{R}^{m\times k}$, 从而实现了对信息的压缩.  

然而这个计算方法往往并没有很好的效果, 其原因在于:
- 在高维情况下计算复杂
- 一些 function words (如 "the", "a", "of" 等含义不大但频繁出现的词) 会对结果产生较大的影响
  - 解决方法: 
    - 对数据进行 scaling, 如对数变换
    - 设置一个阈值, 如 $\text{max}(X_{ij}, t\approx 100)$, 即若过于频繁则将其置为一个固定值
    - 直接忽略

### Pointwise Mutual Information (PMI)

- PMI 最早提出见 [Word Association Norms, Mutual Information, and Lexicography (Church & Hanks, 1990)](https://aclanthology.org/J90-1003.pdf). 并在 [Similairty-based estimation of word co-occurrence probabilities (Dagan et al., 1994)](https://www.aclweb.org/anthology/J94-4003.pdf) 等工作中被广泛用于词汇的 similarity 估计等任务. [Neural Word Embedding as Implicit Matrix Factorization (Omer Levy, Yoav Goldberg)](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf) 中指出, negative sampling 的 skip-gram 模型某种意义上是一种隐式的 PMI.

**下具体介绍其理论原理.**

- PMI 是基于 co-occurrence matrix 的一种改进方法. 给定一个词语 $\mathrm{w}$ 和其上下文词语 $\mathrm{c}$, 其 PMI 定义为:
  $$
  \text{PMI}(\mathrm{w}, \mathrm{c}) = \log \frac{P(\mathrm{w}, \mathrm{c})}{P(\mathrm{w})P(\mathrm{c})}   = \log \frac{\#(\text{w,c} )   |\mathcal{D}| }{{\#(\mathrm w)\#(\mathrm c)  }}
  $$
  其中 $\#(\mathrm w)$ 表示词语 $\mathrm w$ 在语料库 $\mathcal{D}$ 中出现的次数, $\#(\text{w,c})$ 表示词语 $\mathrm w$ 和 $\mathrm c$ 共同出现的次数, $|\mathcal{D}|$ 表示语料库的大小. 
- 因此我们可以使用这个 PMI 来代替 co-occurrence matrix 衡量词语之间的相关性.

### Positive Pointwise Mutual Information (PPMI)
- 根据概率论的知识, 当 $\mathrm w$ 和 $\mathrm c$ 独立时, $\text{PMI}(\mathrm w, \mathrm c) = 0$. 因此 PMI 可以被理解为 $\mathrm w$ 和 $\mathrm c$ 之间的相关性. 若 $\mathrm w$ 和 $\mathrm c$ 的出现频率甚至低于独立情况 (即类似于二者会避免同时出现), 则 PMI 会是负值. 
- PPMI 对 PMI 进行了修正, 使得其只保留正值:
  $$
  \text{PPMI}(\mathrm w, \mathrm c) = \max(\text{PMI}(\mathrm w, \mathrm c), 0)
  $$ 
  这样的操作更加突出了词语之间的正向关连性.

- ***SPPMI (Shifted PPMI)***: 为了进一步改进, 我们可以对 PPMI 进行平滑处理, 使得其不至于过于稀疏. 一种常见的方法是对其进行平滑处理, 如:
  $$
  \text{SPPMI}(\mathrm w, \mathrm c) = \max(\text{PMI}(\mathrm w, \mathrm c) - \log(k), 0)
  $$
  其中 $k$ 是一个平滑参数.

## GloVe: Global Vectors for Word Representation

[GloVe: Global Vectors for Word Representation (Jeffrey Pennington, Richard Socher, Christopher D. Manning)](https://nlp.stanford.edu/pubs/glove.pdf)



### Introduction

GloVe 也是一种 word embedding 方法.  然而与 word2vec 不同, word2vec 是基于一个上下文窗口的预测模型, 而 GloVe (其代表 **Glo**bal **Ve**ctors) 是基于词语的共现信息来学习词向量, 也就是这个 dataset 的全局信息进行学习.

### Co-occurrence Matrix 共现矩阵

GloVe 的核心思想是利用词语的共(同出)现信息来学习词向量, 在数学上可以表示为一个共现矩阵 $X$. 这里重申其定义:
- $X$: co-occurrence matrix, $X_{ij}$ 表示以$i$为中心词, $j$在其上下文中出现的次数
- 第$i$行的行和: $X_i = \sum_{k} X_{ik}$, 表示词$i$的上下文中所有词出现的总次数
- $P_{ij} = \mathbb P(j|i) = \mathbb{P}(j \text{ is the context of } i)= \frac{X_{ij}}{X_i}$: 词$j$在词$i$的上下文中出现的概率

下例是在 GloVe 中的一个例子, 其计算了一个包含 6 billion tokens 的语料库中, 词语 *ice* 和 *steam* 的真实共现概率:
  |Probability \& Ratio| $k = \textit{solid}$ | $k = \textit{gas}$ | $k = \textit{water}$ | $k = \textit{fashion}  \text{ (noise)}$ |
  |---|---|---|---|---|
  | $\mathbb P (k\mid \text{ice})$ |  $1.9 \times 10^{-4}$ | $6.6 \times 10^{-5}$ | $3.0 \times 10^{-3}$ | $1.7 \times 10^{-5}$ |
  | $\mathbb P (k\mid \text{steam})$ |  $2.2 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | $2.2 \times 10^{-3}$ | $1.8 \times 10^{-5}$ |
  | ${\mathbb P (k\mid \text{ice})}/{\mathbb P (k\mid \text{steam})}$ |  $8.9 > 1$ | $0.085 < 1$ | $1.36 \approx 1$ | $0.96 \approx 1$ |

注意关注这个表格中的大小关系及随后一行与$1$的比较. 对于最后一行的比值, 一种理解方法是它代表了单词 $k$ 在词语 *ice* 和 *steam* 的上下文中出现的相对频率 (其实是 odds ratio). 
- 若 $\text{Odds Ratio} > 1$, 则说明 $k$ 与分子词 *ice* 的语义关系更为密切.
- 若 $\text{Odds Ratio} < 1$, 则说明 $k$ 与分母词 *steam* 的语义关系更为密切.
- 若 $\text{Odds Ratio} \approx 1$, 则说明 $k$ 与两个词的语义关系相对平均. 可能是都比较相关 (如这里的 *water* 之于 *ice* 和 *steam*), 也有可能是都不相关 (如这里的 *fashion*).

因此 GloVe 的核心思想是就是通过这个 Odds Ratio 来进行建模, 从而学习词向量.

### GloVe Model

一个直觉的方法就是, 我们试图确定一个函数 $\hat F$ 来拟合这个 Odds Ratio:
$$
\hat F(\mathbf{w}_i, \mathbf{w}_j, \mathbf{\tilde w}_k)  = \frac{\mathbb P (k\mid i)}{\mathbb P (k\mid j)} \quad( \dagger)
$$
其中 $\mathbf{w}_i, \mathbf{w}_j, \mathbf{\tilde w}_k$ 分别是我们想找的词向量. 特别的, 为了方便, 用 $^\sim$ 符号标识当前的 context word.

接下来我们试图确定这个函数的具体形式. 

- 首先,  由于我们希望这个函数能够捕捉到词语之间的关系, 一个自然的想法是将其表示为两个词向量的差. 并且通过内积来表示两个词向量之间的相似性 (且注意到右侧的 Odds Ratio 是一个比值, 因此内积的形式正好也可以使得参与运算的向量的结果是一个比值标量). 因此可以将 $\hat F$ 表示为:
  $$
  \hat F(\mathbf{w}_i, \mathbf{w}_j, \mathbf{\tilde w}_k) = \hat F \left( (\mathbf{w}_i - \mathbf{w}_j)^\top \cdot \mathbf{\tilde w}_k \right)
  $$
- 其次, 为了数学计算的方便, 进一步假设 $\hat F$ 具有 homomorphic property, 即有下式成立:
  $$
  \hat F \left( (\mathbf{w}_i - \mathbf{w}_j)^\top \cdot \mathbf{\tilde w}_k \right) = \frac{\hat F \left( \mathbf{w}_i^\top \cdot \mathbf{\tilde w}_k \right)}{\hat F \left( \mathbf{w}_j^\top \cdot \mathbf{\tilde w}_k \right)}
  $$
- 结合 $\dagger$ 和上一步的 homomorphic 性质, 我们知道: $\hat F \left( (\mathbf{w}_i - \mathbf{w}_j)^\top \cdot \mathbf{\tilde w}_k \right) \stackrel{\dagger}{=} \frac{\mathbb P (k\mid i)}{\mathbb P (k\mid j)} \stackrel{\tiny{\text{Hommph.}}}{=} \frac{\hat F \left( \mathbf{w}_i^\top \cdot \mathbf{\tilde w}_k \right)}{\hat F \left( \mathbf{w}_j^\top \cdot \mathbf{\tilde w}_k \right)}$, 故
   $$\begin{aligned}
  \hat F \left( \mathbf{w}_i^\top \cdot \mathbf{\tilde w}_k \right) = \frac{\hat F \left( \mathbf{w}_j^\top \cdot \mathbf{\tilde w}_k \right)}{\mathbb{P}(k\mid j)}  \mathbb{P}(k\mid i)\triangleq c\cdot  \mathbb{P}(k\mid i)  \quad \diamond
  \end{aligned}$$
  - > 注意到这里通过引入一个常数 $c$, 某种意义上隐藏掉了词汇 $j$ 的影响. 这使得我们从三个词 $\mathbf{w}_i, \mathbf{w}_j, \mathbf{\tilde w}_k$ 的关系中, 重点关注了 $\mathbf{w}_i$ 和 $\mathbf{\tilde w}_k$ 之间的关系.

- 再再进一步, 我们又知道 $\exp(\cdot)$ 就是一个具有上述 homomorphic property 的函数. 因此我们可以将 $F$ 表示为:
  $$
  \hat F \left( \mathbf{w}_i^\top \cdot \mathbf{\tilde w}_k \right) = \exp(\mathbf{w}_i^\top \cdot \mathbf{\tilde w}_k) \quad \star
  $$

- 结合 $\diamond$ 和 $\star$, 我们可以得到 $F \left( \mathbf{w}_i^\top \cdot \mathbf{\tilde w}_k \right) = c\cdot  \mathbb{P}(k\mid i) = \exp(\mathbf{w}_i^\top \cdot \mathbf{\tilde w}_k)$, 从而等式同取对数得到:
  $$
  \mathbf{w}_i^\top \cdot \mathbf{\tilde w}_k = \log(\mathbb{P}(k\mid i)) + \log(c)  = \log(X_{ik}) - \log(X_i) + \log(c)
  $$
  其中 $b$ 是 bias 项, 最后一个等式是由于最开始对于 co-occurrence matrix 的定义: $P_{ij} = \mathbb P(j|i) = \frac{X_{ij}}{X_i}$.

- 最终为了简化, 我们可以将 $\log(c)$ 展开且将 $\log(X_i)$ 等项一并合并到 bias 项中, 从而整理得到一个关于 $X_{ik}$ 的更为简洁的表达式:
  $$
  \mathbf{w}_i^\top \cdot \mathbf{\tilde w}_k  + b_i + \tilde b_k   = \log(X_{ik})  
  $$
  - 这个式子就是 GloVe 的核心表达式. 
    - 式子的右侧 $\log(X_{ik})$ 在训练的角度是我们根据真实的语料库统计出来的 co-occurrence matrix 的真实值; 
    - 式子的左侧是我们根据一个 intuition ($\dagger$) 出发推导出的 $\hat F (\mathbf{w}_i, \mathbf{w}_j, \mathbf{\tilde w}_k)$ 的表达式, 即一个估计值
    - 我们希望通过改变 $\mathbf{w}_i, \mathbf{w}_j, \mathbf{\tilde w}_k$ 来使得估计值尽可能接近真实值. 这和最小二乘法的思想是一致的.

因此经过上面的推导, 我们可以自然的得到 GloVe 的损失函数:
$$
\mathcal{J} = \sum_{i,k=1}^{|V|} f(X_{ik}) \left( \mathbf{w}_i^\top \cdot \mathbf{\tilde w}_k  + b_i + \tilde b_k   - \log(X_{ik}) \right)^2
$$
其中:
- $f(X_{ik})$ 是一个具体形式已知的权重函数, 使得模型更加稳定性能更好. 但其对主线思路的理解影响不大, 故暂不展开.
- $i, k$ 分别是词汇表 $V$ 中的词语索引. $k$ 是 $i$ 的上下文词语, $\mathbf{w}_i, \mathbf{\tilde w}_k$ 分别是词语 $i, k$ 的词向量.
- $b_i, \tilde b_k$ 分别是词语 $i, k$ 的 bias 项.
- $\log(X_{ik})$ 是真实的 co-occurrence matrix 的值.
- $|V|$ 是词汇表的大小.

这个损失函数的优化可以通过 SGD 等方法进行.
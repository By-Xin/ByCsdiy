# Autoregressive Model

## Motivations and Introductions

### Target Goals

- Generating data: Image, Audio, Text, etc.
- Compressing data: entrophy based compression
- Anomaly detection: detect abnormal data (e.g. to correct the predictions of Supervised Learning sometimes when encountering rare data)
  
### Likelihood-Based Generative Models

Assume the real world data follows some distribution $p_{data}$, and we can observe some samples from it:  $x^{(1)}, x^{(2)}, \cdots, x^{(n)} \sim p_{data}$.

The goal is to learn a model $p$ to approximate $p_{data}$ that allows:

- Computing $p(x)$ for any $x$
- Sampling $x \sim p(x)$

Now introduce **Function Approximation**: learn $\theta$ such that $p_\theta$ is a good approximation of $p_{data}$.

### Autoregressive Models

Likelihood-based models that modeling **discrete data**.

We want to estimate distributions of complex, high-dimensional data, such as images, audio, and text.

We also care about computational and statistical efficiency (reasonable amount of data to learn a good model)


## Simple Autoregressive Model: Histograms

***Recall*** 
Our goal is to estimate $p_{data}$ from samples $x^{(1)}, x^{(2)}, \cdots, x^{(n)} \sim p_{data}$.

### Training

![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202311161406277.png)

We train the model by *COUNTING FREQUENCIES*.
For example, there are $n$ samples, and total $k$ different values, then we can estimate the probability of each value by $p_i = \frac{\text{\# of i appears}}{n}~ (i = 1,\cdots ,k)$.

### Inference and Sampling

**Querying** $p_i$ for any $i$ is to lookup into the array $p_1,\cdots,p_k$.

**Sampling** is to choose a random value of the emperical cdf $F_i$ of $p$ and reversely find the corresponding $i$, specifically:

1. Calculate the cumulative distribution function (CDF) 
  $$F_i = \sum_{j=1}^i p_j$$
2. Sample $u \sim U[0,1]$
3. Return $i$ such that $F_{i-1} < u \leq F_i$

***HOWEVER***

- Failure in high dimensions (too many parameters / probabilities to estimate).
- Poor in generalization (overfitting)

## Modern Neural Autoregressive Model

### Fitting Distributions

Generally, we

- Assume that $x^{(1)}, x^{(2)}, \cdots, x^{(n)} $ are sampled from a *real* distribution $p_{data}$.
- Build a model class with parameters $\theta$: $p_\theta$.
- Set a critrion to optimize $\theta$ (a search problem):
  $$
  \arg\min_\theta \text{loss}(\theta,x^{(1)}, x^{(2)}, \cdots, x^{(n)})
  $$

Plus, we want our optimization being able to:

- work with large datasets
- $p_\theta$ is as close as possible to $p_{data}$
- be generalizable to new data

### Estimation 

***Maximum Likelihood Estimation***
$$
\arg\min_\theta \text{loss}(\theta,x^{(1)}, x^{(2)}, \cdots, x^{(n)}) = \arg\min_\theta \sum_{i=1}^n - \log p_\theta(x^{(i)})
$$


***KL Divergence Estimation***

It is equivalent to *minimizing KL divergence* between *Empirical Data* distribution and *Model* distribution:

$$ \hat p_{data}(x) = \frac1n\sum_{i=1}^n \bold{1}\{x^{(i)} = x\}$$
$$ KL(\hat p_{data}||p_\theta) = \mathbb{E}_{x \sim \hat p_{data}} \left[ -\log p_\theta(x) \right] - H(\hat p_{data})$$

where $H(\hat p_{data})$ is the entropy of $\hat p_{data}$, which is a constant.

***Stochastic Gradient Descent***

### Model Designing

Recall that we want to model $p_\theta(x)$, where $x$ is a vector of discrete values, and $p_\theta$ is a probability distribution over $x$. Therefore, we should ensure that 
$$ \text{ for all } \theta:\begin{cases}
p_\theta(x) \geq 0 \\
\sum_{x} p_\theta(x) = 1
\end{cases}
\forall x
$$

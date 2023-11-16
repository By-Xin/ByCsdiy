# 李宏毅-Flow Based Model

## Generative Models Introduction

- Component by Component (Auto-regressive Model)
  - Problem
    - What’s the best order?
    - Slow generation
- Variational Auto-encoder
  - optimizing a lower bound
- Generative Adversarial Network (GAN)
  - Unstable training

## What is Generator

- Generator $G$ is a network, defining a probability distribution (*PDF*) $p_G$.
- General pattern:

    $$
    x = G(z)\\
    z \rightarrow^\mathcal{G} x
    $$

  - $x$ is a *high demensional vector* (goal output)(e.g. picture that we generated), with *complex distribution* $p_G(x)$
  - $z$ sampled from  distribution $\pi(z)$, (e.g. Gaussian distribution)

- The goal of generator is to make $p_G(x)$ as close as possible to the real data distribution $p_{data}(x)$.
  - *USUALLY*
    - Sampling $\{x^{(1)}, x^{(2)}, \dots, x^{(m)}\}$ from $p_{data}(x)$
    - $G^* = \arg\max_G \sum_{i=1}^m \log p_G(x^{(i)}) \approx \arg\min_G KL(p_{data}(x) || p_G(x))$
    - *HOWEVER,* $p_G$ is unkown or very complex, so we can't compute $KL(p_{data}(x) || p_G(x))$ directly
  - ***FLOW BASED MODEL*** can directly optimze the objective function
  
## Math Prerequisite

### Jacobian Matrix

Assume $z = (z_1, z_2, \dots, z_n)$, $x = (x_1, x_2, \dots, x_n)$, $f: \mathbb{R}^n \rightarrow \mathbb{R}^n, f(z) = x$

$$
J_f = \frac{\partial f}{\partial z} = \begin{bmatrix}
\frac{\partial f_1}{\partial z_1} & \frac{\partial f_1}{\partial z_2} & \dots & \frac{\partial f_1}{\partial z_n}\\
\frac{\partial f_2}{\partial z_1} & \frac{\partial f_2}{\partial z_2} & \dots & \frac{\partial f_2}{\partial z_n}\\
\vdots & \vdots & \ddots & \vdots\\
\frac{\partial f_n}{\partial z_1} & \frac{\partial f_n}{\partial z_2} & \dots & \frac{\partial f_n}{\partial z_n}\\
\end{bmatrix}
$$

$$
J_{f^{-1}} = \frac{\partial z}{\partial f} = \begin{bmatrix}
\frac{\partial z_1}{\partial f_1} & \frac{\partial z_1}{\partial f_2} & \dots & \frac{\partial z_1}{\partial f_n}\\
\frac{\partial z_2}{\partial f_1} & \frac{\partial z_2}{\partial f_2} & \dots & \frac{\partial z_2}{\partial f_n}\\
\vdots & \vdots & \ddots & \vdots\\
\frac{\partial z_n}{\partial f_1} & \frac{\partial z_n}{\partial f_2} & \dots & \frac{\partial z_n}{\partial f_n}\\
\end{bmatrix}
$$

We can proof that:
$$J_{f^{-1}} J_f = I$$

***E.g.***

$$
f(z) = \begin{bmatrix}
f_1(z_1, z_2)\\
f_2(z_1, z_2)
\end{bmatrix} = \begin{bmatrix}
z_1 + z_2\\
2z_1
\end{bmatrix}
$$

$$
J_f = \begin{bmatrix}
\frac{\partial f_1}{\partial z_1} & \frac{\partial f_1}{\partial z_2}\\
\frac{\partial f_2}{\partial z_1} & \frac{\partial f_2}{\partial z_2}
\end{bmatrix} = \begin{bmatrix}
1 & 1\\
2 & 0
\end{bmatrix}
$$

### Determinant

$$ \det(A) = 1/\det(A^{-1})$$

$$ \det(J_{f^{-1}}) = 1/\det(J_f)$$

- Determinant of matrix $A$ is the volume of the parallelepiped spanned by the row vectors of $A$.

### Change of Variable Theorem

***Assume*** we have *INPUT* $z$, with a distribution $\pi(z)$; a function $f: \mathbb{R}^n \rightarrow \mathbb{R}^n$; *OUTPUT* $x$ :$f(z) = x$; $x$'s distribution $p(x)$.

***E.g.***

$$ z \sim U(0, 1), x = f(z) = 2z+1 \\ \Rightarrow x \sim U(1, 3)$$

Given that

$$ \int p(x) dx = 1, \int \pi(z) dz = 1$$

Then we have

$$ p(x) = \frac12 \pi(z)$$

***More Generally :***

- Given point $z'$ from distribution $\pi(z')$, $x'$ from distribution $p(x')$, and $f$ is known.
- Take little volume $\Delta z$ around $z'$ to get $(z', z'+\Delta z)$ , and accordingly $(x', x'+\Delta x)$.
- Since $\Delta z$ and $\Delta x$ are small, we can assume that $(z', z'+\Delta z)$ is *UNIFORMLY DISTRIBUTED*, and so does $(x', x'+\Delta x)$.
- Moreover, since $x$ is transformed from $z$, from $(z', z'+\Delta z)$ to $(x', x'+\Delta x)$, the uniform distribution should be of the same volume
  - i.e. $$p(x')\Delta x = \pi(z')\Delta z $$ $$\boxed{p(x') = \pi(z') \frac{\Delta z}{\Delta x} = \pi(z')  \left|\frac{\partial z}{\partial x}\right|}$$
  - As long as $f$ is given, $\frac{\partial z}{\partial x}$ is fixed, so we can get $p(x)$ from $\pi(z)$

***Multi-dimensionally:***

$$\boxed{p(x) = \pi(z) \left|\det\left(\frac{\partial z}{\partial x}\right)\right| = \pi(z) \left|\det\left(J_{f^{-1}}\right)\right|}$$

## Flow Based Model Introduction

### Ready for FLOW

***Recall*** the goal of generator:
$$
G^* = \arg\max_G \sum_{i=1}^m \log p_G(x^{(i)}),
$$

while
$$
p_G(x^{(i)}) = \pi(z^{(i)}) \left|\det\left(J_{G^{-1}}\right)\right|.
$$

Since $x^{(i)}  = G(z^{(i)})$, we can get $z^{(i)} = G^{-1}(x^{(i)})$.
Thus,
$$
\log p_G(x^{(i)}) = \log \pi(G^{-1}(x^{(i)})) + \log \left|\det\left(J_{G^{-1}}\right)\right|.
$$

***Task awaits:***

- Compute $\det(J_G)$
- How to solve $G^{-1}$ (as it could be very large and computationally expensive)

  - To ensure the invertibility, there are some constraints:
    - In FLOW, we assume that $x,z$ are of the ***SAME DIMENSION***
    - $G$ is limited to be ***INVERTIBLE***

### Intuitions of  FLOW

To compensate the limitation of $G$, we can have multiple layers of $G$'s (Just like a flow of water!), which is:
$$
\pi(z) \stackrel{\mathcal{G_1}}{\rightarrow} p_1(x) \stackrel{\mathcal{G_2}}{\rightarrow} p_2(x) \stackrel{\mathcal{G_3}}{\rightarrow} \dots \stackrel{\mathcal{G_n}}{\rightarrow} p_n(x)
$$
$$
z^{(i)} = G_1^{-1}(x^{(i)})\left(\cdots G_k^{-1}(x^{(i)})\right)
$$
Thus we have:
$$
\begin{aligned}
p_1(x^{(i)}) &= \pi(z^{(i)}) \left|\det\left(J_{G_1^{-1}}\right)\right|\\
p_2(x^{(i)}) &= p_1(z^{(i)}) \left|\det\left(J_{G_2^{-1}}\right)\right| = \pi(z^{(i)}) \left|\det\left(J_{G_1^{-1}}\right)\right| \left|\det\left(J_{G_2^{-1}}\right)\right|\\
\cdots\\
p_k(x^{(i)}) &= \pi(z^{(i)}) \left|\det\left(J_{G_1^{-1}}\right)\right| \left|\det\left(J_{G_2^{-1}}\right)\right| \dots \left|\det\left(J_{G_k^{-1}}\right)\right|
\end{aligned}
$$

Therefore, the MLE becomes:
$$
\log p_k(x^{(i)}) = \log \pi(z^{(i)}) + \sum_{j=1}^k \log \left|\det\left(J_{G_j^{-1}}\right)\right|
$$

### Calculations of FLOW

#### Start with only 1 generator

$$
z \stackrel{\mathcal{G}}{\rightarrow} x
$$

Here,  
$$
\arg\max_G \log p_G(x^{(i)}) = \arg\max_G\left(\log \pi(G^{-1}(x^{(i)})) + \log \left|\det\left(J_{G^{-1}}\right)\right|\right).
$$

> We can see that, originally, $G$ is a network that maps $z \stackrel{\mathcal{G}}{\rightarrow} x$, but for training we need to compute $G^{-1}$.

In order to maximize the objective function, we need to maximize $\log \pi(G^{-1}(x^{(i)}))$ and $\log \left|\det\left(J_{G^{-1}}\right)\right|$  as much as possible.
First, view them respectively:

- $\log \pi(G^{-1}(x^{(i)}))$:
  - Presumably $\pi(x)$ is a simple distribution (i.e. Gaussian distribution here)
  - For a (standard) normal distribution $\pi(x)$, $x=0$ is the maximum point.
  - Thus we want to make $G^{-1}(x^{(i)})$ as close to $0$ as possible.
  - YET obviously there's something wrong with this as we don't want to squeeze all $z$'s to $0$ vectors.

- $\log \left|\det\left(J_{G^{-1}}\right)\right|$:
  - Note that, if $z^{(i)} \equiv 0$, then $J_{G^{-1}} $ would be zero matrix, its determinant would be $0$, and thus $\log \left|\det\left(J_{G^{-1}}\right)\right|$ would be $-\infty$.

Thus, we can see that there exists a tradeoff between the parts. And that's what's going on in FLOW.

The following content will show us how to calculate specifically.

#### Coupling Layer

***Design of Coupling Layer***
  
  Assume that we have input vector $z = (z_1, ..., z_D)'$ and output vector $x = (x_1, ..., x_D)'$. 

  We can split them into two parts, $z_a := (z_1,...,z_d), z_b:=(z_{d+1},...,z_D)$, and $x_a := (x_1,...,x_d), x_b:=(x_{d+1},...,x_D)$.

  - For the first part of output ($x_a$), just copy the input: $(x_1,...,x_d)' = (z_1,...,z_d)'$.
  
  - For the second part of output ($x_b$):
    - Introduce a function $\mathcal{F}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{D-d}$, where $\mathcal{F}\left((z_1,...,z_d)'\right) := (\beta_{d+1},..., \beta_D)'$.
    - Introduce another function $\mathcal{H}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{D-d}$, where $\mathcal{H}\left((z_1,...,z_d)'\right) := (\gamma_{d+1},..., \gamma_D)'$.
    ***Note that $\mathcal{F}$ and $\mathcal{H}$ can be any functions without restrictions, and thus can be however complex (i.e. CNN, Deep Network, etc.)***
    - We can calculate it as
      $$\begin{aligned}
      x_{b} &= (x_{d+1},...,x_D)
      \\&= \mathcal{F}(z_a) \odot z_b + \mathcal{H}(z_a)
      \\&= (z_{d+1},...,z_D)' \odot (\beta_{d+1},..., \beta_D)' + (\gamma_{d+1},..., \gamma_D)' 
      \end{aligned}$$ where $\odot$ is the element-wise product.

![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202311160043424.png)

***Using Coupling Layer to assist MLE***

Recall that our goal is:
$$
\arg\max_G \log p_G(x^{(i)}) = \arg\max_G\left(\log \pi(G^{-1}(x^{(i)})) + \log \left|\det\left(J_{G^{-1}}\right)\right|\right).
$$

**For the first part** $\log \pi(G^{-1}(x^{(i)}))$, coupling method allows us to calculate $G^{-1}(x^{(i)})$ easily. From the structure of coupling layer, we can see that:
$$z_{i\le d} = x_i $$ $$ z_{i>d} = \frac{x_i - \mathcal{H}(x_a)}{\mathcal{F}(x_a)} = \frac{x_i - \gamma_i}{\beta_i}$$

**For the second part** $\log \left|\det\left(J_{G^{-1}}\right)\right|$, we have to focus on the Jacobian matrix $J_G$. Similarly, we can split $J_G$ into four parts:
$$
J_G = \begin{bmatrix}
J_{aa} & J_{ab}\\
J_{ba} & J_{bb}
\end{bmatrix}
$$
where $J_{aa}$ is the Jacobian matrix of $x_a$ w.r.t. $z_a$, $J_{ab}$ is the Jacobian matrix of $x_a$ w.r.t. $z_b$, and so on.
  ![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202311160039118.png)
- $J_{aa}$ is an identity matrix, as $x_a = z_a$.
- $J_{ab}$ is a zero matrix, as $x_a$ is independent of $z_b$ (as $x_a$ is completely copied from $z_a$).
- $J_{ba}$ is trivial, as for this Jacobian Matrix, the final goal is to calculate the determinant, which will not be influence by the value of $J_{ba}$ given that $J_{ba}$ is a zero matrix and $J{aa}$ is an identity matrix.
- $J_{bb}$ is the most important part. Actually it is a diagonal matrix.
  - $\left(x_{d+1},...,x_D\right)' = \left(z_{d+1},...,z_D\right)' \odot \left(\beta_{d+1},..., \beta_D\right)' + \left(\gamma_{d+1},..., \gamma_D\right)' \Leftrightarrow x_{i>d} = \beta_iz_{i>d}+\gamma_i$, which shows that only in the diagonal part of $J_{bb}$, the value is non-zero.
  
To sum up, the overall determinant of $J_G$ is:
$$
\det J_G = \frac{\partial x_{d+1}}{\partial z_{d+1}} \frac{\partial x_{d+2}}{\partial z_{d+2}} \dots \frac{\partial x_{D}}{\partial z_{D}} = \beta_{d+1}\beta_{d+2}\dots\beta_D
$$

#### Stacking: From 1 to FLOW

![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202311160058950.png)

- In stacking part, we may find that a simple copy-and-paste strategy is not appliable, as it will cause the fact that the first half of $x$ will be the direct copy of the original input $z$, which is the Gaussian noise.

- To solve this problem, as is shown in the figure above, we choose to copy and paste different parts of $z$ to different parts of $x$ in a flow - not sticking to the first half of $x$ in all layers.
  - Specifically, for image generation, we can either chose to split the image *by pixels*, or *by channels*.


### GLOW

***1$\times$1 Convolution***

![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202311160812627.png)

Here we introduce a shuffle matrix $W\in\mathbb R^{3\times3}$. We take out each pixcel of input $z$ for all layers, and then multiply it with $W$ to get the output $x$.
For example, assume that 
$$
W = \begin{bmatrix}
0 & 0 & 1\\
1 & 0 & 0\\
0 & 1 & 0
\end{bmatrix}
$$
and
$$
z_1 = \begin{bmatrix}
1 \\2\\3 
\end{bmatrix}
$$
then
$$
x_1 = Wz_1 =  \begin{bmatrix}
3 \\1\\2
\end{bmatrix}.
$$

Here, we can see that with matrix $W$ shuffling, we can directly copy and paste the first half of the splited input in FLOW. And now $W$ will work as a *learnable* matrix, which can be trained to get a better shuffle.

As $W$ is actually a $G$ function, thus if $W$ is invertible, then we can get $W^{-1}$.

As for the determinant of Jacobian matrix, we can see that, for $x = f(z) = Wz$,$J_f = W$. Thus the Jacobian matrix for GLOW is:
![](https://michael-1313341240.cos.ap-shanghai.myqcloud.com/202311160829565.png)

THUS, we can see that the determinant of Jacobian matrix is:
$$
\det J_f = (\det W)^{d\times d}
$$

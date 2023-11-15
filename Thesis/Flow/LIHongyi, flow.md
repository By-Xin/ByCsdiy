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

## Flow Based Model

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

To ensure the invertibility, there are some constraints:
  - In FLOW, we assume that $x,z$ are of the ***SAME DIMENSION***
  - $G$ is limited to be ***INVERTIBLE***

To compensate the limitation of $G$, we can have multiple layers of $G$'s (Just like a flow of water!), which is:
$$
\pi(x) \stackrel{\mathcal{G_1}}{\rightarrow} p_1(x) \stackrel{\mathcal{G_2}}{\rightarrow} p_2(x) \stackrel{\mathcal{G_3}}{\rightarrow} \dots \stackrel{\mathcal{G_n}}{\rightarrow} p_n(x)
$$


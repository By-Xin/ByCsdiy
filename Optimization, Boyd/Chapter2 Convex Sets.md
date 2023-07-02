# Chapter 2: Convex Sets

## 2.1 Affine and Convex Sets

### Affine Sets 仿射集
- 定义1：一个集合是仿射集，若$\forall x_1 , x_2 \in c $， 则连接 $x_1 , x_2 $ 的直线也在这个集合内
- 定义2: 设$x_1 \cdots x_k \in c , \theta_1 , \ldots , \theta_k \in R , \sum \theta_i = 1$；对于一个仿射集，从中任意选择$k$个点构成的仿射组合$\theta_1 x_1 + \cdots + \theta_k x_k \in c$也成立
- 事实上，这两个定义是等价的
  - $[p.f]~def_1 \Rightarrow def_2$ 
    - 已有仿射集$c$,$x_1 , x_2 , x_3 \in c$ , $\theta_1+\theta_2+\theta_3 = 1$
    - 则$\frac{\theta_1}{\theta_1 + \theta_2}x_1 + \frac{\theta_2}{\theta_1 + \theta_2}x_2 \in c $
    - 进一步$(\theta_1+\theta_2)(\frac{\theta_1}{\theta_1 + \theta_2}x_1 + \frac{\theta_2}{\theta_1 + \theta_2}x_2) + \theta_3 x_3 = \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 \in c$ 
  - $def_2 \Rightarrow def_1$是显然的
  
- Affine Set 的性质
  -  定义AffineSet $C$ 的一个subspace $V$
     -  if $x_0 \in C, V := C-x_0 = \{x-x_0 | x \in C\}$, then $V$ is a subspace，这里称V为*与C相关的子空间*
  - Subspace 具有性质：$\forall \alpha, \beta, v_1 , v_2 \in V, \alpha v_1 + \beta v_2 \in V$
  - 这里的$x_0$是任意的； 同时由于这是一个空间，因此一定是经过原点的

- 定理：任意线性方程组$Ax = b$的解集是仿射集（反之亦然）
  - pf：设$x_1 , x_2$是$Ax = b$的解，则c是仿射集等价于$\theta_1 x_1 + \theta_2 x_2 \in c$；又由于c是线性方程组的解集，故上式等价于$A(\theta_1 x_1 + \theta_2x_2)=b$，等价于$\theta b+(1-\theta)b \equiv b$ ,Q.E.D
  - 相似地构造subspace:
    $$V = \{x- x_0 | x \in C\}, \forall x_0 \in C \\ = \{x-x_0 | Ax = b\} , Ax_0 = b\\ =  \{x-x_0 | A(x-x_0) = 0\} \\= \{y | Ay = 0\}$$

- 任意集合的仿射集的构造：
  - 引入affine hull（仿射包）的定义：
    - $\operatorname{aff} C = \{\theta_1 x_1 + \cdots + \theta_k x_k | x_1 \cdots x_k \in C, \theta_1 + \cdots + \theta_k = 1\}$
  - 这个仿射包就是由任意集合能够构造的最小的仿射集 

### Convex Sets 凸集

- 定义：一个集合C是凸集，当任意两点之间的线段都在C内，即
$$\forall x_1 , x_2 \in C, \theta \in [0,1], \theta x_1 + (1-\theta)x_2 \in C$$
  - *仿射集一定是一个凸集*
- 类似仿射组合，定义**凸组合**：
  - $\theta_1 x_1 + \cdots + \theta_k x_k \in C, \theta_1 + \cdots + \theta_k = 1, \theta_i \in [0 ,1]$

- 类似def2的等价命题：*C为Convex Set $\Leftrightarrow$ 任意元素的凸组合属于C*

- 类似仿射包，定义**凸包**:
  - $\operatorname{Conv}C = \{\theta_1 x_1 + \cdots + \theta_k x_k | x_1 \cdots x_k \in C, \theta_1 + \cdots + \theta_k = 1, \theta_i \in [0,1]\}$


### Cones 锥, Convex Cones 凸锥

- Cones定义：$\forall x \in C, \theta \ge 0 , \theta x\in C$,
- Convex Cones:  $\forall x_1 ,x_2 \in C, \theta_1 , \theta_2 \ge 0 , x_1\theta_1+x_2\theta_2 \in C$
  - *凸锥一定经过原点*
- 类似定义**凸锥组合**：$\theta_1 x_1 + \cdots + \theta_k x_k \in C, \theta_i \ge 0$
- 类似定义**凸锥包**：$ \{\theta_1 x_1 + \cdots + \theta_k x_k | x_1 \cdots x_k \in C, \theta_i \ge 0\}$
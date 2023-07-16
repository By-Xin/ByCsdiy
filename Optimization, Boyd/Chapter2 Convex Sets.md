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

## 2.2 几种重要的凸集

1. $\R^n$ 空间
2. $\R^n$ 空间的子空间
3. 任意直线
4. 任意线段
5. $\{x_0+\theta v | \theta \ge 0\}, x_0 \in \R^n, v \in \R^n$
6. ...

### Hyperplane 超平面 与 Halfspaces 半空间

- Hyperplane def：$H = \{x | a^Tx = b\}, a \ne 0$
  - 一个超平面一定是一个仿射集，但是不一定是一个凸锥（i.i.f.超平面经过原点）
- Halfspaces def：$H = \{x | a^Tx \le b\}, a \ne 0$ 
  - 粗略地看，超平面将一个空间分割成了两个半空间
  - 半空间是一个凸集，但不是一个仿射集，不一定是一个凸锥（i.i.f.半空间包含原点）

### Euclidean balls 与 Ellipsoids 欧氏空间的球与椭球

- Euclidean Balls def: $B = \{x ~|~ \|x-x_c\|_2 \le r\}$
  - 一个欧氏球是一个凸集，但不是一个仿射集，不是一个凸锥（除非退化成原点）

  *p.f. Euclidean Balls is a Convex Set*:
    - $\forall x_1 , x_2 \in B \Rightarrow ||x_1 - x_c||_2 \le r, ||x_2 - x_c||_2 \le r$
    - p.f. Convex Set $ \Leftrightarrow \forall \theta \in [0,1], ||\theta x_1 + (1-\theta)x_2 - x_c||_2 \le r$
    - $$ LHS \equiv ||\theta (x_1 - x_c)+(1-\theta)(x_2 - x_c)||_2 \\ \le ||\theta (x_1 - x_c)||_2 + ||(1-\theta)(x_2 - x_c)||_2 (三角不等式) \\ \le \theta r + (1-\theta)r = r$$ 

- Ellipsoids def: $ \Epsilon(x_c, P) = \{x ~|~ (x-x_c)^TP^{-1}(x-x_c) \le 1\}$
  - 其中$P \in \S_{++}^n$，即P是一个n阶对称正定矩阵
    - 正定矩阵def：奇异值均大于0
      - 奇异值def：$A$的奇异值是$A^TA$的特征值的平方根
    - 事实上，P的每个奇异值描述了这个椭球的一个半轴长



# Statoinary Time Series

## 1. General Linear Process

***Definition*** *(General linear process):*
$$
Y_t =  \sum _{j=0}^{\infty} \psi_j \epsilon_{t-j}.
$$ where $\epsilon_t$ is a white noise process with mean 0 and variance $\sigma^2$. To ensure the sum converges, we require that $\sum_{j=0}^{\infty} \psi_j^2 < \infty$.

For GLM, an special case is expontential decay.

***Definition*** *(Expontential decay):*
$$
Y_t = \sum _{j=0}^{\infty} \phi ^j \epsilon_{t-j}.
$$ where $|\phi| < 1$.

In this case, we have:
>$$
\begin{aligned}
&\mathbb{Var}(Y_t) = \frac{\sigma^2}{1-\phi^2},\\
&\mathbb{Cov}(Y_t, Y_{t+h}) = \phi^h \frac{\sigma^2}{1-\phi^2},\\
&\mathbb{Corr}(Y_t, Y_{t+h}) = \phi^h.
\end{aligned}
$$

More generally, for GLM, we have:
>$$
\begin{aligned}
&\gamma_h = \mathbb{Cov}(Y_t, Y_{t+h}) =\sigma^2 \sum_{j=0}^{\infty} \psi_j \psi_{j+h}.
\end{aligned}
$$

## 2. Moving Average Process

***Definition*** *(Moving average process):*
$$
Y_t = \epsilon_t - \theta_1 \epsilon_{t-1} - \theta_2 \epsilon_{t-2} - \cdots - \theta_q \epsilon_{t-q}.
$$ where $\epsilon_t$ is a white noise process with mean 0 and variance $\sigma^2$.

### 2.1 MA(1) Process

$$
Y_t = \epsilon_t - \theta \epsilon_{t-1}.
$$

In this case, we have:
>$$
\begin{aligned}
&γ_0 = \sigma ^2 (1 + \theta^2),\\
&γ_1 = -\sigma^2 \theta,\\
&\rho_1 = \frac{-\theta}{1+\theta^2},\\
&\rho_h = 0, \quad h \geq 2.
\end{aligned}
$$

Thus when time lag $h \geq 2$, the correlation is 0.

***Properties***:
- When $\theta = -1$, $\max \rho_h = 1/2$; when $\theta = 1$, $\min \rho_h = -1/2$.
- It can be proved that, replacing $\theta$ with $1/\theta$, the correlation function is the same.

### 2.2 MA(2) Process

$$
Y_t = \epsilon_t - \theta_1 \epsilon_{t-1} - \theta_2 \epsilon_{t-2}.
$$

In this case, we have:
>$$
\begin{aligned}
&γ_0 = \sigma ^2 (1 + \theta_1^2 + \theta_2^2),\\
&γ_1 = -\sigma^2 (-\theta_1 + \theta_1 \theta_2),\\
&γ_2 = -\sigma^2 \theta_2,\\
&\rho_1 = \frac{\theta_1 + \theta_2}{1 + \theta_1^2 + \theta_2^2},\\
&\rho_2 = \frac{\theta_1 \theta_2}{1 + \theta_1^2 + \theta_2^2},\\
&\rho_h = 0, \quad h \geq 3.
\end{aligned}
$$

### 2.3 MA(q) Process

$$
Y_t = \epsilon_t - \theta_1 \epsilon_{t-1} - \theta_2 \epsilon_{t-2} - \cdots - \theta_q \epsilon_{t-q}.
$$

In this case, we have:
>$$
\begin{aligned}
&\gamma_0 = (1+\theta_1^2 + \cdots + \theta_q^2)σ^2,\\
&\rho_h = \frac{ -\theta_h + \theta_1\theta_{h+1} + \cdots + \theta_{q-h}\theta_q}{1+\theta_1^2 + \cdots + \theta_q^2}, (h = 1, 2, \cdots, q),\\
&\rho_h = 0, \quad h \geq q+1.
\end{aligned}
$$

*Note:*
- **It would *CUT OFF* at lag $q$.**
- MA process, as a special case of GLM, is always ***linear, stationary, and causal***.

The characteristic polynomial of MA(q) process is:
$$
\theta(x) = 1 - \theta_1 x - \theta_2 x^2 - \cdots - \theta_q x^q.
$$
And MA(q) process can be written as:
$$
Y_t = \theta(B) \epsilon_t = (1 - \theta_1 B - \theta_2 B^2 - \cdots - \theta_q B^q) \epsilon_t.
$$ where $ B $ is the backshift operator.

## 3. Autoregressive Process

***Definition*** *(Autoregressive process):*

$$
Y_t = \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \epsilon_t.
$$ where $\epsilon_t$ is a white noise process with mean 0 and variance $\sigma^2$, and is independent of historical information: $\epsilon_t \perp Y_{t-1}, Y_{t-2}, \cdots, Y_{t-p}$.

Its characteristic polynomial is:
$$
\phi(x) = 1 - \phi_1 x - \phi_2 x^2 - \cdots - \phi_p x^p.
$$
And the AR(p) process can be written as:
$$
\phi(B)Y_t = \epsilon_t.
$$ where $ B $ is the backshift operator.

***Stationarity***:

MA(q) process is always stationary, but AR(p) process is not always stationary. The ***stationarity condition*** of AR(p) process is that all the roots of the characteristic polynomial are outside the unit circle, i.e. its roots are $|x_i| > 1$.

If AR(p) process is stationary, then it has a ***unique stationary solution*** (i.e. a unique set of $\psi$'s in the following equation), with the form of ***MA(∞)***:
$$
Y_t = \psi_0 \epsilon_t + \psi_1 \epsilon_{t-1} + \cdots = \sum_{j=0}^{\infty} \psi_j \epsilon_{t-j}.
$$ where $|\phi| < 1$.

*Solving AR(p) process is to find the coefficients $\psi$ in the MA(∞) representation.*

### 3.1 AR(1) Process

$$
Y_t = \phi Y_{t-1} + \epsilon_t \quad ~†
$$

#### Autocovariance and Autocorrelation

- First, calculate the **variance** by applying $var$ to both sides of $†$:
$$
\begin{aligned}
\gamma_0 = \phi^2 \gamma_0 + \sigma^2,\\
⇒ \gamma_0 = \frac{\sigma^2}{1-\phi^2}.
\end{aligned}
$$

    Here, we have an extra condition that $|\phi| < 1$.

- Then, calculate the **covariance** by duplicating $Y_{t-k}$ on both sides of $†$ and taking expectation:
$$
\begin{aligned}
\mathbb{E}(Y_t Y_{t-k}) &= \phi \mathbb{E}(Y_{t-1} Y_{t-k}) + \mathbb{E}(\epsilon_t Y_{t-k}), \\
\end{aligned}
$$ adding that $\mathbb{E}(ϵ_t Y_{t-k}) = 0$, thus:
$$
\begin{aligned}
\gamma_k = \phi \gamma_{k-1}.
\end{aligned}
$$
Recursively, we have:
$$
\begin{aligned}
\gamma_k = \phi^k \gamma_0 =  \phi^k\frac{\sigma^2}{1-\phi^2}.
\end{aligned}
$$

- Finally, calculate the **correlation**:
$$
\begin{aligned}
\rho_k = \frac{\gamma_k}{\gamma_0} = \phi^k.
\end{aligned}
$$

To conclude, for AR(1) process, we have:
>$$
\begin{aligned}
&\gamma_0 = \frac{\sigma^2}{1-\phi^2},\\
&\gamma_k = \phi^k \gamma_0,\\
&\rho_k = \phi^k.
\end{aligned}
$$

***Properties***:
- As $|\phi| < 1$, the correlation function is decreasing exponentially. If $0 < \phi < 1$, the correlation is positive; if $-1 < \phi < 0$, the correlation goes interchanging between positive and negative.

#### MA(∞) Representation of AR(1) Process

For AR(1) process, we can have an infinite MA representation:
$$
Y_t = \sum_{j=0}^{\infty} \phi^j \epsilon_{t-j}.
$$ where $|\phi| < 1$.

#### Stationarity of AR(1) Process

*For AR(1) process, iif $|\phi| < 1$, the process is stationary.*

### 3.2 AR(2) Process

$$
Y_t = \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \epsilon_t.
$$

Introduce AR characteristic polynomial:
$$
\phi(x) = 1 - \phi_1 x - \phi_2 x^2.
$$  and accordingly its characteristic equation:
$$
\phi(x) =  1- \phi_1 x - \phi_2 x^2 = 0.
$$

#### Stationarity of AR(2) Process

*For AR(2) process, ( if $e_t$ is independent of $Y_{t-1}, Y_{t-2}$ ), the process is stationary iif the roots of the characteristic equation are outside the unit circle, i.e. its roots are $|x_1| > 1$ and $|x_2| > 1$.*

A sufficient and necessary condition for stationarity in AR(2) process is that:
$$
\phi_1+\phi_2 < 1, \quad \phi_2 - \phi_1 < 1, \quad |\phi_2| < 1.
$$


#### Autocovariance and Autocorrelation

First, by **multiplying $Y_{t-k}$ on both sides** of the AR(2) process and **taking expectation**, we have:
$$
γ_k = \phi_1 γ_{k-1} + \phi_2 γ_{k-2}.
$$

Then, we can calculate the **autocorrelation**:
$$
ρ_k = \phi _1 ρ_{k-1} + \phi_2 ρ_{k-2}, ~ k = 1, 2, \cdots.
$$

It is also called the ***Yule-Walker equation***.

Plus the initial conditions $ρ_0 = 1, \quad ρ_{-1} = \rho _1$, we can solve the equation:
$$
\begin{aligned}
    \rho_1 &= \frac{\phi_1}{1-\phi_2}
\\ \rho_2 &= \phi_1 \rho_1 + \phi_2 ρ_0 = \frac{\phi_2(1-\phi_1^2)+\phi_1^2}{1-\phi_2}
\end{aligned}
$$

- ***For AR process, the ACF is exponentially decreasing in any cases.***

After solving ACF, we can calculate the variance by multiplying $Y_t$ on both sides of the AR(2) process and taking expectation:
$$
\begin{aligned}
    \gamma_0 &= \phi_1 \gamma_1 + \phi_2 \gamma_2 + \sigma^2\\
    &= \phi_1 \rho_1γ_0 + \phi_2 \rho_2γ_0 + \sigma^2 \\
\end{aligned}
$$thus 
$$
 γ_0 = \frac{\sigma^2}{1-\phi_1^2 - \phi_2^2}~.
$$


### 3.3 AR(p) Process

$$
Y_t = \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \epsilon_t.
$$

#### Stationarity of AR(p) Process

Its characteristic polynomial is:
$$
\phi(x) = 1 - \phi_1 x - \phi_2 x^2 - \cdots - \phi_p x^p.
$$

By solving the characteristic equation, we can get the roots of the equation, and the ***stationarity condition*** is that ***all the roots are outside the unit circle,** i.e. $|x_i| > 1$*.

One necessary but not sufficient condition is that:
$$
\phi_p + \phi_{p-1} + \cdots + \phi_1 < 1, \quad |\phi_p| < 1.
$$

#### Yule-Walker Equations

By multiplying $Y_{t-k}$ on both sides of the AR(p) process, taking expectation, and dividing by $\gamma_0$, we have the general form of ***Yule-Walker equations***:
$$
\begin{aligned}
    \rho_k &= \phi_1 \rho_{k-1} + \phi_2 \rho_{k-2} + \cdots + \phi_p \rho_{k-p}, \quad k = 1, 2, \cdots, p.
\end{aligned}
$$

Expanding it into equations, we have:
$$
\begin{aligned}
    \rho_1 &= \phi_1 + \phi_2 \rho_1 + \cdots + \phi_p \rho_{p-1},\\
    \rho_2 &= \phi_1 \rho_1 + \phi_2 + \cdots + \phi_p \rho_{p-2},\\
     &\quad\quad\cdots \\
    \rho_p &= \phi_1 \rho_{p-1} + \phi_2 \rho_{p-2} + \cdots + \phi_p.
\end{aligned}
$$

Concluding it into matrix form:
$$
\begin{aligned}
    \begin{bmatrix}
        \rho_1 \\
        \rho_2 \\
        \vdots \\
        \rho_p
    \end{bmatrix}
    &=
    \begin{bmatrix}
        \rho_0 & \rho_1 & \cdots & \rho_{p-1} \\
        \rho_1 & \rho_0 & \cdots & \rho_{p-2} \\
        \vdots & \vdots & \ddots & \vdots \\
        \rho_{p-1} & \rho_{p-2} & \cdots & \rho_0
    \end{bmatrix}
    \begin{bmatrix}
        \phi_1 \\
        \phi_2 \\
        \vdots \\
        \phi_p
    \end{bmatrix}.
\end{aligned}
$$  and can be denoted as $ϱ = \mathcal{R}\varPhi$.

### 3.4 Solving AR Process *(AR(p) $\to$ MA(∞))*

***AR(1)***

$$
Y_t = \theta_0 + \phi Y_{t-1} + \epsilon_t.
$$

Bringing stationary solution 
$$
Y_t = \mu + \sum _{j=0}^{\infty} \psi_j \epsilon_{t-j}.
$$ to both sides ($Y_t$ and $Y_{t-1}$), we have:
$$
\mu + \sum _{j=0}^{\infty} \psi_j \epsilon_{t-j} = \phi_0 + \phi_1 (\mu + \sum _{j=0}^{\infty} \psi_j \epsilon_{t-1-j}) + \epsilon_t.
$$
By comparing the coefficients of each $\epsilon_t$, we have:
$$
\begin{aligned}
    \mu &= \theta_0 + \phi \mu,\\
    \psi _0 &= 1,\\
    \psi _1 &= \phi \psi_0 ,\\
    \psi _k &= \phi \psi_{k-1}, \quad k \geq 2.
\end{aligned}
$$
Thus the stationary solution for AR(1) process is:
$$
Y_t = \frac{\theta_0}{1-\phi} + \sum _{j=0}^{\infty} \phi^j \epsilon_{t-j}.
$$

***AR(p)***

Similarly, we have:
$$
\begin{aligned}
    \mu &= \theta_0 + \phi_1 \mu + \phi_2 \mu + \cdots + \phi_p \mu,\\
    \psi _0 &= 1,\\
    \psi _1 &= \phi_1 \psi_0 ,\\
    \psi _k &= \phi_1 \psi_{k-1} + \phi_2 \psi_{k-2} + \cdots + \phi_p \psi_{k-p}, \quad k \geq 2.
\end{aligned}
$$


## 4. ARMA Process

***Definition*** *(ARMA process):*

$$
Y_t = \phi_1 Y_{t-1} + \cdots + \phi_p Y_{t-p} + \epsilon_t - \theta_1 \epsilon_{t-1} - \cdots - \theta_q \epsilon_{t-q}.
$$ where $\epsilon_t$ is a white noise process with mean 0 and variance $\sigma^2$.

It can be written as:
$$
\phi(B)Y_t = \theta_0 + \theta(B)\epsilon_t.
$$ where $ B $ is the backshift operator.

Here, we have AR characteristic polynomial $\phi(x)$ and MA characteristic polynomial $\theta(x)$:
$$
\begin{aligned}
    \phi(x) &= 1 - \phi_1 x - \cdots - \phi_p x^p,\\
    \theta(x) &= 1 - \theta_1 x - \cdots - \theta_q x^q.
\end{aligned}
$$

When AR characteristic equation has all roots outside the unit circle, the model has ***unique stationary solution***. 

### 4.1 Autocovariance and Autocorrelation


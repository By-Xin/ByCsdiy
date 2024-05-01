# Statoinary Time Series

## General Linear Process

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

## Moving Average Process

***Definition*** *(Moving average process):*
$$
Y_t = \epsilon_t - \theta_1 \epsilon_{t-1} - \theta_2 \epsilon_{t-2} - \cdots - \theta_q \epsilon_{t-q}.
$$ where $\epsilon_t$ is a white noise process with mean 0 and variance $\sigma^2$.

### MA(1) Process

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

### MA(2) Process

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

### MA(q) Process

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

It would *cut off* at lag $q$.


## Autoregressive Process

***Definition*** *(Autoregressive process):*

$$
Y_t = \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \cdots + \phi_p Y_{t-p} + \epsilon_t.
$$ where $\epsilon_t$ is a white noise process with mean 0 and variance $\sigma^2$, and is independent of historical information: $\epsilon_t \perp Y_{t-1}, Y_{t-2}, \cdots, Y_{t-p}$.

### AR(1) Process

$$
Y_t = \phi Y_{t-1} + \epsilon_t \quad ⋯~†
$$

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
- 
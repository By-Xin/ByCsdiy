# Ordered Sets and Boundedness

## Rational Numbers & Real Numbers

### **Rational Numbers** $\mathbb{Q}$

- Rational numbers have some problems:
  - *Algebraic Incompleteness*: Can write equations in $\mathbb{Q}$; can't find solutions in $\mathbb{Q}$.
    - E.g. $x^2 = 2$.
  - *Analytic Incompleteness*: Can find a sequence of rational numbers that converges to an irrational number.
    - E.g. $1, 1.4, 1.41, 1.414, \ldots$ converges to $\sqrt{2}$.
    - *Claim: Let $A=\{p\in \mathbb{Q}|p>0,p^2<2\}$, then $\sup A$ does not exist in $\mathbb{Q}$.*
      - Proof: Let $q = p - \frac{p^2-2}{p+2}$, then it is easy to show that $q>p$. 
      - We need to show that $q^2<2$. Consider $q^2-2 = \frac {2(p^2-2)^2}{(p+2)^2}$. Thus $q^2 = 2+\frac{2(p^2-2)^2}{(p+2)^2} < 2$.
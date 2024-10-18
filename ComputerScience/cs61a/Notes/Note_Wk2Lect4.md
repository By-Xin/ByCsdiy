# Note_Wk2Lect4

## Higher-Order Functions

- **Naming and functions** allow us to **abstract** away a vast amount of complexity. While each function definition has been trivial, the computational process set in motion by our evaluation procedure is quite intricate.
- It is only by virtue of the fact that we have an extremely general evaluation procedure for the Python language that small components can be composed into complex processes. Understanding the procedure of interpreting programs allows us to validate and inspect the process we have created.
- Benefits of higher-order functions:
  - Express general methods of computation
  - Remove repetition from programs
  - Separate concerns among functions

### Functions as Arguments

```python
def identity(x):
    return x

def square(x):
    return x * x

def summation(n, term):
    total, k = 0, 1
    while k <= n:
        total, k = total + term(k), k + 1
    return total

def sum_naturals(n):
    return summation(n, identity)

def sum_squares(n):
    return summation(n, square)
```

### Functions as Returns

*Example*: Make adder, which returns a function that takes an argument `x` and returns `x + n`.

```python
def make_adder(n):
    >>> add_three = make_adder(3) # add_three is a function: def add_three(x): return x + 3
    >>> add_three(4)
    7
    def adder(x): # `adder` is local def statement, which can refer to the enclosing scope
        return x + n
    return adder
```

- Function inside a function is called a **closure**. A closure is a function that captures free variables in its enclosing scope.

### Lambda Expressions

*Example*

```python
square = lambda x: x * x
square(4) # 16
```

- Lambda has no `return`
- Lambad function must be a single expression

### Return Statements

- `return` statement completes the evaluation of the function call and provides the value of the function call.
- `f(x)` for user-defined function `f`: switch to a new environment, execute the body of `f` with `x` bound to the argument value
- `return` statement within `f`: switch back to the environment where `f` was called(the previous environment), and the value of the function call is the value of the `return` statement.

### Control Statements

```python
# For statement below, either `if` or `else` branch is executed, but not both
from math import sqrt
def real_sqrt(x):
    if x > 0:
        return sqrt(x)
    else:
        return 0.0

# Yet for our own `if_` function, both branches are evaluated, and this may cause an error
def if_(condition, true_result, false_result):
    if condition:
        return true_result
    else:
        return false_result

def _real_sqrt(x):
    return if_(x > 0, sqrt(x), 0.0) 

```

### Conditional expressions

- python will first evaluate the <predicate>
- if the <predicate> is true, then the <consequent> is evaluated and returned
- if the <predicate> is false, then the value of the whole expression is the value of the <alternative>

```python
<consequent> if <predicate> else <alternative>
```

*Example*

```python
abs(1/x if x != 0 else 0)
```

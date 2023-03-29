# Questions

## Boolean Values

- why `False or 0` equals `0`?
  - Ans: Becasue `or` will first check if *LHS* is *TRUE*. If so, it will return `True`. Else, it will then check the *RHS* and return *RHS*

- why `1/0 or True` will return ERROR?
  - Ans: Because `or` will first check if *LHS* is *TRUE*. But while checking, it will meet the computational error, and the comparing process will be suspended and return ERROR.
  - To compare with, `True or 1/0` will directly return `True`. Since `or` statement first find `True` on LHS, it will not have furture process, and thus `1/0` will not be actually executed.


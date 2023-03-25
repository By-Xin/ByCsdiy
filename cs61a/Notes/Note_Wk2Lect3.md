## Doc Test Model in Python

* syntax:

  ```python
  def func_divide(N,D):
  '''
      >>> q,r = func_divide(2013,10)
      >>> q
      201
      >>> r
      3
  '''
      return N//D, N%D
  
  ```
  
* While adding a doc test in Python functions, the system will automatically run the code quoted in`'''` , and will then compare the results that you listed here and the computer returns. If everything goes well, it will remain silent, but if the demo program fails, it will return warnings. It is a good idea to add such help documentation in functions, in order to pretest the functions.

## Default values in function

* syntax:

```python
    def func_divide(N,D=10):
'''
        >>> q,r = func_divide(2013,10)
        >>> q
        201
        >>> r
        3
'''
        return N//D, N%D
```

* Usage:
  * If there is no value assigned to D, then D will be bounded with the default value 10.


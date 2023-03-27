# Note: Lab01

## Function's return 

- When Python executes a `return` statement, the function terminates immediately. If Python reaches the end of the function body without executing a `return` statement, it will automatically return `None`.

- Notice also that `print` will display text **without the quotes**, but `return` will preserve the quotes.

  ```python
  def what_prints():
      print('Hello World!')
      return 'Exiting this function.'
      print('61A is awesome!')
  
  >>> what_prints()
  Hello World!
  'Exiting this function.'
	```



## Boolean Values

### TRUE or FALSE in python

- In python, those negative integers also mean TRUE if not specifially defined. (which is different from the basic settings in the lecture of DataStructure)

- Those represent *FALSE*: `0`, `None`,`False`

  - *Please also note that the spelling of `False`, ( rather than `FALSE` in R )*

- Example:

  ```python
  positive = -9
  negative = -12
  while negative:
      if positive:
          print(negative)
      positive += 3
      negative += 3
  '''
  expected output:
  -12
  -9
  -6
  '''
  ```

### Boolean Operations

```python
>>> True and False
False
>>> True or False
True
>>> not False
True
```

- To evaluate the expression `<left> and <right>`:
  1. If `<left>` *FALSE* then *FALSE*
  2. Else `<right>` 


- To evaluate the expression `<left> or <right>`:
  1. If `<left>` *TRUE* then *TRUE*
  2. Else `<right>`

Functions that perform comparisons and return boolean values typically begin with `is`, not followed by an underscore (e.g., `isfinite`, `isdigit`, `isinstance`, etc.).
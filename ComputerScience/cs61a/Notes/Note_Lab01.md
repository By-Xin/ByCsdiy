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



## Debugging Techniques

### Traceback Messages

The **first** line in such a pair has the following format:

```
File "<file name>", line <number>, in <function>
```

- **File name**: the name of the file that contains the problem.
- **Number**: the *line number* in the file that caused the problem, or the line number that contains the next function call
- **Function**: the *name of the function* in which the line can be found.

### Error Messages

The very **last** line in the traceback message is the error statement. 

```
<error type>: <error message>
```

- **Error type**: the type of error that was caused (e.g. `SyntaxError`, `TypeError`). These are usually descriptive enough to help you narrow down your search for the cause of error.
- **Error message**: a more detailed description of exactly what caused the error. Different error types produce different error messages.

### Tech1: Running doctests

Python has a great way to quickly write tests for your code. These are called doctests, and look like this:

```python
def foo(x):
    """A random function.

    >>> foo(4)
    4
    >>> foo(5)
    5
    """
```

The lines in the docstring that look like interpreter outputs are the **doctests**. 

### Tech2: Writing your own tests

- Write some tests before you write code
- Write more tests after you write code
- Test edge cases

### Tech3: Using `print` statements

- Don't just print out a variable -- add some sort of message to make it easier for you to read:

  ```python
  print(tmp)   # harder to keep track
  print('DEBUG: tmp was this:', tmp)  # easier
  ```

- Use `print` statements to view the results of function calls (i.e. after function calls).

- Use `print` statements at the end of a `while` loop to view the state of the counter variables after each iteration:

  ```python
  i = 0
  while i < n:
      i += func(i)
      print('DEBUG: i is', i)
  ```

- Don't just put random `print` statements after lines that are obviously correct.

#### Long-term debugging

To contro whether entring such a debugging mode, use a global `debug` variable:

```python
debug = True

def foo(n):
i = 0
while i < n:
    i += func(i)
    if debug:
        print('DEBUG: i is', i)
```

Now, whenever we want to do some debugging, we can set the global `debug` variable to `True`, and when we don't want to see any debugging input, we can turn it to `False` (such a variable is called a *"flag"*).

### Tech4: Using `assert` statements

`assert` statement lets you *test that a condition is true*, and print an error message otherwise in a single line. 

```python
def double(x):
    assert isinstance(x, int), "The input to double(x) must be an integer"
    return 2 * x
```


One *major* benefit of assert statements is that they are more than a debugging tool, you can leave them in code permanantly. A key principle in software development is that it is generally better for code to crash than produce an incorrect result, and having asserts in your code makes it far more likely that your code will crash if it has a bug in it.
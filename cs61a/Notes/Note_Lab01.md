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




## TRUE or FALSE in python

- In python, those negative integers also mean TRUE if not specifially defined. (which is different from the basic settings in the lecture of DataStructure)

- Please remember that only those '0's will represent FALSE. 

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
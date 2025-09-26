'''1.	Write a function to calculate factorial of a number.
2.	Write a function that returns the sum of all elements in a list.
3.	Write a program that prints the multiplication table of a number.
4.	Write a function that checks if a word is a palindrome.
5.	Implement a simple calculator with functions for add, subtract, multiply, divide.'''

#1.	Write a function to calculate factorial of a number
def calc_factorial(num):
    fact =1
    for i in range(1,num+1):
        fact  = fact * i

    return fact

print(calc_factorial(5))

#2. Write a function that returns the sum of all elements in a list
def get_sum(nums):
    return sum(nums)

print("Sum of numbers [1,4,7] --> ",get_sum([1,4,7]))

#3. Write a program that prints the multiplication table of a number.
def print_multpltable(num):
    for i in range(1,11):
        print(f"{num} X {i} = {num*i}")

print_multpltable(7)

#4. Write a function that checks if a word is a palindrome.
import math as m

def is_palindrome(val):
    rangetocheck  = m.floor(len(val)/2) # If odd number of characters then we dont need to verify the middle character
    palcheck = True
    for i in range(rangetocheck):
        if val[i] != val[len(val)-(i+1)]:
            palcheck = False
            break

    return palcheck

word = 'tattarrattat'
print(is_palindrome(word))


#5.	Implement a simple calculator with functions for add, subtract, multiply, divide.'''

class mymath:

    def __init__(self,a,b):
        self.a = a
        self.b = b
    
    def add(self): 
        print (f"addtion of {self.a} and {self.b} = {self.a + self.b}")

    def substract(self):
        print (f"substraction of {self.a} and {self.b} = {self.a - self.b}")

    def multiply(self):
        print (f"multiplication of {self.a} and {self.b} = {self.a * self.b}")

    def divide(self):
        print (f"division of {self.a} and {self.b} = {self.a / self.b}")


mymathclass = mymath(8,2)
mymathclass.add()
mymathclass.substract()
mymathclass.multiply()
mymathclass.divide()
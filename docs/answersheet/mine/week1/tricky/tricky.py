'''
1.	Write a recursive function to compute Fibonacci numbers.
2.	Implement a class Rectangle with methods to compute area and perimeter.
3.	Write a program that simulates a simple bank account (deposit, withdraw, balance).
4.	Implement a function that removes duplicates from a list without using set().
5.	Write a custom iterator class that yields squares of numbers up to N.
''' 

#1.	Write a recursive function to compute Fibonacci numbers.
def get_fibonacci(n:int) -> int:
    if n<0:
        raise ValueError("Fibonacci series is not defined for negative numbers")
    if n==0:
        return 0
    if n==1:
        return 1
    return get_fibonacci(n-1)+get_fibonacci(n-2)

#2.	Implement a class Rectangle with methods to compute area and perimeter.
class Rectangle:

    def __init__(self,length,width):
        self.length=length
        self.width=width

    def area(self):
        return self.length * self.width
    
    def perimeter(self):
        return 2 * (self.length + self.width)

#3.	Write a program that simulates a simple bank account (deposit, withdraw, balance).
class Mnrbank:

    acctnumbers  =[]
    currbalance =0

    def openacct(self,bankacctnumber):
        if not isinstance(bankacctnumber,int):
            return "Please provide a valid bank account number"
        elif bankacctnumber in Mnrbank.acctnumbers:
            return "Account number already in use"
        else:
            Mnrbank.acctnumbers.append(bankacctnumber)
            return f"Bank Account numbers {bankacctnumber} successfully opened"
        
    def getbalance(self):
        return Mnrbank.currbalance
    
    def depositamount(self,depostiamt):
        if not isinstance(depostiamt,int):
            return "\n Please provide a valid deposit amount"
        else:
            Mnrbank.currbalance+=depostiamt
            return f"\n your current balance after the deposit of ${depostiamt} is --> {Mnrbank.currbalance}"
        
    def withdrawamount(self,withdrawamt):
        if not isinstance(withdrawamt,int):
            return "\n Please provide a valid withdrawl amount"
        elif withdrawamt > Mnrbank.currbalance:
            return f"\n INSUFFICIENT BALANCE . -->  Your current balance is. {Mnrbank.currbalance} and you are trying to withdraw. {withdrawamt}"
        else:
            Mnrbank.currbalance-=withdrawamt
            return f"\n your current balance after the withdrawl of ${withdrawamt} is --> {Mnrbank.currbalance}"
        

#4.	Implement a function that removes duplicates from a list without using set().
from typing import List

def getdistinctitems(listtocheck:List[int]) -> List[int]:
    newlist =[]

    for item in listtocheck:
        if item not in newlist:
            newlist.append(item)

    return newlist

#5.	Write a custom iterator class that yields squares of numbers up to N.
class SquaresUpTo:
    """Custom iterator yielding squares of numbers up to N (inclusive)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.current = 1

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.current > self.n:
            raise StopIteration
        result = self.current ** 2
        self.current += 1
        return result


if __name__ == "__main__":
    #1
    print("\nFibonacci values of 6 is. --> ",get_fibonacci(6))

    #2
    rectclass = Rectangle(4,6)
    print("\n Area of rectange --> ",rectclass.area())
    print("\n Perimeter of rectange --> ",rectclass.perimeter())

    #3
    mybankacct = Mnrbank()
    print("\n",mybankacct.openacct(1234))
    print("\n You current balance for the account '1234'. --> ",mybankacct.getbalance())
    print(mybankacct.depositamount(123))
    print(mybankacct.withdrawamount(23))
    
#4
print(f"\n original list [1,2,5,7,9,3,6,2,4,9] \n distint list. --> {getdistinctitems([1,2,5,7,9,3,6,2,4,9])}")

#5
squares = list(SquaresUpTo(5))  # [1,4,9,16,25]
print(squares)

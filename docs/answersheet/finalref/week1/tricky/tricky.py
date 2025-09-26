# Week 1 — Tricky Mini Coding Challenges (Implemented Reference)

#### Mini Coding Challenges
'''
1.	Write a recursive function to compute Fibonacci numbers.
2.	Implement a class Rectangle with methods to compute area and perimeter.
3.	Write a program that simulates a simple bank account (deposit, withdraw, balance).
4.	Implement a function that removes duplicates from a list without using set().
5.	Write a custom iterator class that yields squares of numbers up to N.
'''

from typing import List


def fib_recursive(n: int) -> int:
    """Return the nth Fibonacci number using recursion.
    0th = 0, 1st = 1, 2nd = 1, etc.
    """
    if n < 0:
        raise ValueError("Fibonacci is undefined for negative indices")
    if n in (0, 1):
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


class Rectangle:
    """Rectangle with methods to compute area and perimeter."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


class BankAccount:
    """Simple bank account simulation."""

    def __init__(self, balance: float = 0.0) -> None:
        self.balance = balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount

    def get_balance(self) -> float:
        return self.balance


def remove_duplicates(lst: List[int]) -> List[int]:
    """Remove duplicates from a list while preserving order."""
    seen = []
    for item in lst:
        if item not in seen:
            seen.append(item)
    return seen


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
    # Fibonacci
    assert fib_recursive(0) == 0
    assert fib_recursive(5) == 5

    # Rectangle
    r = Rectangle(3, 4)
    assert r.area() == 12
    assert r.perimeter() == 14

    # BankAccount
    acc = BankAccount(100)
    acc.deposit(50)
    acc.withdraw(30)
    assert acc.get_balance() == 120

    # Remove duplicates
    assert remove_duplicates([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]

    # SquaresUpTo
    squares = list(SquaresUpTo(5))  # [1,4,9,16,25]
    assert squares == [1, 4, 9, 16, 25]
# Week 1 — Medium Mini Coding Challenges (Implemented Reference)

from typing import List


def factorial(n: int) -> int:
    """Return n! (factorial of n). Raise ValueError for negative inputs."""
    if n < 0:
        raise ValueError("Factorial is undefined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def sum_of_list(nums: List[int]) -> int:
    """Return the sum of all elements in the list."""
    return sum(nums)


def multiplication_table(n: int, upto: int = 10) -> List[str]:
    """Return multiplication table lines for n, from 1 up to `upto`."""
    return [f"{n} x {i} = {n * i}" for i in range(1, upto + 1)]


def is_palindrome(word: str) -> bool:
    """Return True if the word is a palindrome, False otherwise."""
    # Normalize case and spaces if needed
    cleaned = word.lower().replace(" ", "")
    return cleaned == cleaned[::-1] #Reversed the string and comparing two values


class Calculator:
    """Simple calculator with add, subtract, multiply, and divide."""

    @staticmethod
    def add(a: float, b: float) -> float:
        return a + b

    @staticmethod
    def subtract(a: float, b: float) -> float:
        return a - b

    @staticmethod
    def multiply(a: float, b: float) -> float:
        return a * b

    @staticmethod
    def divide(a: float, b: float) -> float:
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b


if __name__ == "__main__":
    # Quick self-tests
    assert factorial(5) == 120
    assert sum_of_list([1, 4, 7]) == 12
    assert multiplication_table(3)[0] == "3 x 1 = 3"
    assert is_palindrome("tattarrattat") is True
    calc = Calculator()
    assert calc.add(8, 2) == 10
    assert calc.subtract(8, 2) == 6
    assert calc.multiply(8, 2) == 16
    assert calc.divide(8, 2) == 4.0
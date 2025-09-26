# Week 1 — Easy Mini Coding Challenges (Implemented)

def hello_world() -> str:
    """Return the string 'Hello, World!'."""
    return "Hello, World!"


def greet_name_age(name: str, age: int) -> str:
    """Return a friendly message using name and age."""
    return f"Hello '{name}'! You are {age}."


def max_of_two(a: int, b: int) -> int:
    """Return the larger of a and b."""
    return a if a > b else b  # or: return max(a, b)


def swap(a, b):
    """Return a tuple with the values swapped (b, a)."""
    return b, a


def even_or_odd(n: int) -> str:
    """Return 'Even' if n is even, otherwise 'Odd'."""
    return "Even" if n % 2 == 0 else "Odd"


if __name__ == "__main__":
    # quick sanity checks
    assert hello_world() == "Hello, World!"
    assert greet_name_age("Mahi", 30) == "Hello 'Mahi'! You are 30."
    assert max_of_two(3, 5) == 5
    assert swap(3, 4) == (4, 3)
    assert even_or_odd(7) == "Odd"
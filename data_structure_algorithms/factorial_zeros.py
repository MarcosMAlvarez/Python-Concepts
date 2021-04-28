"""
Find the lower factorial with a given number of zeros
"""
from typing import Any


def zeros(number: int) -> int:
    """
    We get a zero per factor of 5
    """
    num_zeros = 0
    while number:
        num_zeros += number // 5
        number //= 5
    return num_zeros


def linear_search(number: int) -> Any:
    """
    Get the factorial through a linear search
    """
    num_zeros = 0
    while zeros(num_zeros) < number:
        num_zeros += 1

    if number == zeros(num_zeros):
        return num_zeros
    # No solution
    return None


def binary_search(number_zeros: int) -> Any:
    """
    Get the factorial through a binary search
    """
    left = 0
    right = 5 * number_zeros
    while left < right:
        middle = (left + right) // 2
        if zeros(middle) < number_zeros:
            left = middle + 1
        else:
            right = middle

    if zeros(left) == number_zeros:
        return left
    # No solution
    return None

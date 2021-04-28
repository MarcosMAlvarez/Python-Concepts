"""
Find the maximum sum subarray
using three different algorithms
"""


def three_loops(array: list) -> int:
    """
    Uses three loops to get the max sum subarray
    """
    lenght = len(array)
    max_sum = array[0]
    for start in range(lenght):
        for end in range(start, lenght):
            current_sum = sum(array[start : end + 1])
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum


def two_loops(array: list) -> int:
    """
    Uses two loops to get the max sum subarray
    """
    lenght = len(array)
    max_sum = array[0]
    for start in range(lenght):
        current_sum = 0
        for end in range(start, lenght):
            current_sum += array[end]
            if current_sum > max_sum:
                max_sum = current_sum
    return max_sum


def one_loop(array: list) -> int:
    """
    Uses one loop to get the max sum subarray
    """
    lenght = len(array)
    max_sum = array[0]
    current_sum = array[0]
    for i in range(1, lenght):
        current_sum = max(current_sum + array[i], array[i])
        max_sum = max(max_sum, current_sum)
    return max_sum

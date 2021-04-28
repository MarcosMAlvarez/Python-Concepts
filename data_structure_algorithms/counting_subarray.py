"""
Efficient counting subarray with a given sum
"""


def naive(array: list, _sum: int) -> int:
    """Naive implementation"""
    count = 0
    for i in range(len(array)):
        for j in range(i, len(array)):
            if sum(array[i : j + 1]) == _sum:
                count += 1
    return count


def efficient(array: list, _sum: int) -> int:
    """Efficient implementation"""
    count = 0
    prefix_sum_counts = {0: 1}
    current_sum = 0
    # pylint: disable=pointless-string-statement
    """
    current_sum[i] - current_sum[k-1] == s
    => array[k] + ... + array[i] == s
    => count how many are equal to current_sum[i] - s
       at each step
    """
    for numb in array:
        current_sum += numb
        if current_sum - _sum in prefix_sum_counts:
            count += prefix_sum_counts[current_sum - _sum]
        if current_sum in prefix_sum_counts:
            prefix_sum_counts[current_sum] += 1
        else:
            prefix_sum_counts[current_sum] = 1
    return count

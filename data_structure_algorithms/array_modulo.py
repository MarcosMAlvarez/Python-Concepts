"""
Array problem involving modulos
"""


def in_n_cube(array: list, k: int) -> int:
    """Get minimum lenght subarray with sum % number == 0"""
    # pylint: disable=invalid-name
    n = len(array)
    min_len = n + 1
    for i in range(n):
        for j in range(i, n):
            if sum(array[i : j + 1]) % k == 0 and j - i + 1 < min_len:
                min_len = j - i + 1

    return min_len


def efficient(array: list, number: int) -> int:
    """Get minimum lenght subarray with sum % number == 0 efficiently"""
    array_lenght = len(array)
    min_len = array_lenght + 1
    last_prefix_idx = {0: -1}
    current_sum = 0
    for i in range(array_lenght):
        current_sum = (current_sum + array[i]) % number
        if current_sum in last_prefix_idx:
            lenght = i - last_prefix_idx[current_sum]
            if lenght < min_len:
                min_len = lenght
            last_prefix_idx[current_sum] = i

    return min_len


print(in_n_cube([1, 2, 9, 3, 4, 1, 2], 4))
print(efficient([1, 2, 9, 3, 4, 1, 2], 4))

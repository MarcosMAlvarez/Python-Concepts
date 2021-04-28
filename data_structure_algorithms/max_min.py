"""
Find the given-lenght subarray with the
maximum minimum using deque
"""
from collections import deque


def trivial(array: list, lenght: int) -> int:
    """Get the maximum minimum using a trivial solution"""
    maximum = min(array[:lenght])

    for i in range(len(array) - lenght + 1):
        minimum = min(array[i : lenght + i])
        maximum = max(maximum, minimum)

    return maximum


def with_deque(array: list, lenght: int) -> int:
    """Get the maximum minimum using a deque"""
    maximum = min(array[:lenght])
    deq = deque()  # type: deque
    deq.append(0)

    # pylint: disable=consider-using-enumerate
    for i in range(len(array)):
        if i - deq[0] == lenght:
            deq.popleft()
        while bool(len(deq)) and array[deq[-1]] >= array[i]:
            deq.pop()
        deq.append(i)
        if i >= lenght:
            maximum = max(maximum, array[deq[0]])

    return maximum

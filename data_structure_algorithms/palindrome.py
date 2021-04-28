"""
Rolling hashes for constructing a palindrome
"""
from typing import Optional


def naive(string: str) -> Optional[str]:
    """Construct a palindrome in a naive way"""
    str_rev = string[::-1]
    for i in range(len(string), -1, -1):
        print((str_rev[:i], str_rev[i - 1 :: -1]))
        if str_rev[:i] == str_rev[i - 1 :: -1]:
            return string + str_rev[i:]
    return None


# pylint: disable=invalid-name
def rolling_hashes(string: str) -> str:
    """Construct a palindrome with rolling hashes"""
    p = 23
    P = 666013
    str_rev = string[::-1]
    f_foward = 0
    f_backward = 0
    p_power = 1
    max_suffix_palindrome = 0
    for index, char in enumerate(str_rev):
        char_idx = ord(char) - ord("a")
        f_foward = (f_foward + char_idx * p_power) % P
        f_backward = (f_backward * p + char_idx) % P
        p_power *= p
        if f_foward == f_backward:
            max_suffix_palindrome = index
    return string + str_rev[max_suffix_palindrome + 1 :]

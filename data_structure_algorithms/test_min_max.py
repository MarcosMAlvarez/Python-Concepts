"""
Test for min_max.py
"""
from unittest import TestCase
from typing import Tuple
from random import randint

from ..max_min import trivial, with_deque


class TestsMinMax(TestCase):
    """Test class"""

    @staticmethod
    def create_array() -> Tuple[list, int]:
        """Create random array and lenght to test with_deque function"""
        array = [randint(1, 100) for _ in range(20)]
        lenght = randint(1, 10)
        return (array, lenght)

    def test_trivial(self):
        """Test trivial solution"""
        self.assertEqual(trivial([1, 2, 3, 4], 3), 2)
        self.assertEqual(trivial([1, -2, 3, 4], 2), 3)

    def test_with_deque(self):
        """Test with deque solution using random array
        and checks with trivial solution"""
        # Test the function 100 times
        for _ in range(100):
            array, lenght = TestsMinMax.create_array()
            self.assertEqual(with_deque(array, lenght), trivial(array, lenght))

"""
Tests for array_modulo.py
"""
from unittest import TestCase
from random import randint
from typing import Tuple

from ..array_modulo import in_n_cube, efficient


class TestsArrayModulo(TestCase):
    """Test class"""

    @staticmethod
    def create_array() -> Tuple[list, int]:
        """Create random array and lenght to test with_deque function"""
        array = [randint(1, 50) for _ in range(200)]
        lenght = randint(1, 10)
        return (array, lenght)

    def test_n_cube(self):
        """Test trivial solution"""
        self.assertEqual(in_n_cube([1, 2, 9, 3, 1, 2], 4), 2)

    def test_efficient(self):
        """Test with efficient solution using random array
        and checks with trivial solution"""
        # Test the function 10 times
        for _ in range(10):
            array, lenght = TestsArrayModulo.create_array()
            self.assertEqual(efficient(array, lenght), in_n_cube(array, lenght))

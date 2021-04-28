"""
Tests for counting_subarray.py
"""
from unittest import TestCase
from typing import Tuple
from random import randint

from ..counting_subarray import naive, efficient


class TestCS(TestCase):
    """Test class"""

    @staticmethod
    def create_array() -> Tuple[list, int]:
        """Create random arrays for testing"""
        array = [randint(-20, 20) for _ in range(randint(1, 25))]
        number = randint(1, 10)
        return (array, number)

    def test_naive(self):
        """Test naive implementation"""
        self.assertEqual(naive([1, 6, -1, 4, 2, -6], 5), 3)
        self.assertEqual(naive([1, 6, -1, 4, 2, -6], 10), 1)
        self.assertEqual(naive([1, 6, -1, 4, 2, -6], 20), 0)

    def test_efficient(self):
        """Test efficient implementation"""
        for _ in range(10):
            array, _sum = TestCS.create_array()
            self.assertEqual(efficient(array, _sum), naive(array, _sum))

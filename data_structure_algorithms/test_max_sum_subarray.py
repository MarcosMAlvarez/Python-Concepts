"""
Test the functions of max_sum_subarray-py
"""
from unittest import TestCase

from ..max_sum_subarray import three_loops, two_loops, one_loop


class TestMaxSumSubarray(TestCase):
    """Test class"""

    def test_three_loops(self):
        """Test three loops function"""
        self.assertEqual(three_loops([1, 2, 3, -1]), 6)
        self.assertEqual(three_loops([6, 6, -1, -3]), 12)
        self.assertEqual(three_loops([-1, -2, -3]), -1)

    def test_two_loops(self):
        """Test three loops function"""
        self.assertEqual(two_loops([1, 2, 3, -1]), 6)
        self.assertEqual(two_loops([6, 6, -1, -3]), 12)
        self.assertEqual(two_loops([-1, -2, -3]), -1)

    def test_one_loops(self):
        """Test three loops function"""
        self.assertEqual(one_loop([1, 2, 3, -1]), 6)
        self.assertEqual(one_loop([6, 6, -1, -3]), 12)
        self.assertEqual(one_loop([-1, -2, -3]), -1)

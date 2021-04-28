"""
Tests the functions of factorial_zeros.py
"""
from unittest import TestCase

from ..factorial_zeros import binary_search, zeros, linear_search


class TestsFactorialZeros(TestCase):
    """Test class"""

    def test_zeros(self):
        """test zero function"""
        self.assertEqual(zeros(2), 0)
        self.assertEqual(zeros(7), 1)
        self.assertEqual(zeros(10), 2)

    def test_linear_search(self):
        """Test linear search function"""
        self.assertEqual(linear_search(2), 10)
        self.assertEqual(linear_search(3), 15)
        self.assertEqual(linear_search(5), None)

    def test_binary_search(self):
        """Test binary search function"""
        self.assertEqual(binary_search(2), 10)
        self.assertEqual(binary_search(3), 15)
        self.assertEqual(binary_search(5), None)

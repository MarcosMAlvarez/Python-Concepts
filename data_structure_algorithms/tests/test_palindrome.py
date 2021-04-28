"""
Tests for palindrome.py
"""
from unittest import TestCase
from random import randint

from ..palindrome import naive, rolling_hashes


class TestsPalindrome(TestCase):
    """Test class"""

    @staticmethod
    def create_word() -> str:
        """Creates a random word"""
        word = "".join([chr(randint(97, 122)) for _ in range(randint(1, 10))])
        return word

    def test_naive(self):
        """Test naive implementation"""
        self.assertEqual(naive(""), "")
        self.assertEqual(naive("solos"), "solos")
        self.assertEqual(naive("solara"), "solaralos")

    def test_rolling_hashes(self):
        """Test rolling hashes implementation"""
        for _ in range(10):
            word = TestsPalindrome.create_word()
            self.assertEqual(rolling_hashes(word), naive(word))

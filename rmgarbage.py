# -*- coding: utf-8 -*-
# @Author: Tasdik Rahman
# @Date:   2016-03-12
# @Last Modified by:   Tasdik Rahman
# @Last Modified time: 2016-03-25 20:00:56
# @MIT License
# @http://tasdikrahman.me
# @https://github.com/prodicus

"""
Tries to remove strings obtained from OCR engines which are garbage.
An implementation of the paper
'Automatic Removal of “Garbage Strings” in OCR Text: An Implementation'
- by Kazem Taghva , Tom Nartker , Allen Condit , Julie Borsack
References
==========
[1] http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.81.8901
"""


__author__ = "Tasdik Rahman"
__email__ = "prodicus@outlook.com"
__version__ = "0.0.1"
__title__ = 'pyRmgarbage'
__license__ = 'MIT'
__copyright__ = 'Copyright 2016 Tasdik Rahman'
import re
import requests
from bs4 import BeautifulSoup
#import urllib2


class Rmgarbage(object):

    def __init__(self):
        pass

    def too_long(self, string):
        """
        Rule L
        ======
        tests whether the string passed is more than 40 characters, if
        yes. It is garbage!
        :param string: string to be tested
        :returns: either True or False
        """
        return True if len(string) > 40 else False

    def bad_alnum_ratio(self, string):
        """
        Rule A
        ======
        if a string's ratio of alphanumeric characters to total characters is
        less than 50%, the string is garbage
        :param string: string to be tested
        :returns: either True or False
        """

        # matches [^A-Za-z0-9] (^ = not, _ is required)
        pattern = re.compile('[\W_]+')
        alnum_thresholds = {1: 0,     # single chars can be non-alphanumeric
                            2: 0,     # so can doublets
                            3: 0.32,  # at least one of three should be alnum
                            4: 0.24,  # at least one of four should be alnum
                            5: 0.39}  # at least two of five should be alnum

        threshold = alnum_thresholds[len(string)] \
            if len(string) in alnum_thresholds else 0.5

        if len(string) == 0:  # avoid division by zero
            return True
        if float(len(
                pattern.sub('', string)))/len(string) < threshold:
            return True

        return False

    def consecutive_four_identical(self, string):
        """
        Rule R
        ======
        if a string has 4 identical characters in a row, it is garbage
        :param string: string to be tested
        :returns: either True or False
        """
        pattern = re.compile(
            r'((.)\2{3,})')  # matches any 4 consecutive characters
        if pattern.search(string):
            return True

        return False

    def bad_consonant_vowel_ratio(self, string):
        """
        Rule V
        ======
        if a string has nothing but alphabetic characters, look at the
        number of consonants and vowels. If the number of one is less than 10%
        of the number of the other, then the string is garbage.
        This includes a length threshold.
        :param string: string to be tested
        :returns: either True or False
        """
        #alpha_string = filter(str.isalpha, string)
        alpha_string = list(filter(str.isalpha, string))
        vowel_count = sum(1 for char in alpha_string if char in 'aeiouAEIOU')
        consonant_count = len(alpha_string) - vowel_count

        if (consonant_count > 0 and vowel_count > 0):
            ratio = float(vowel_count)/consonant_count
            if (ratio < 0.1 or ratio > 10):
                return True
        elif (vowel_count == 0 and consonant_count > len('rhythms')):
            return True
        elif (consonant_count == 0 and vowel_count > len('IEEE')):
            return True

        return False

    def has_two_distinct_puncts_inside(self, string):
        """
        Rule P
        ======
        Strip off the first and last characters of a string. If there
        are two distinct punctuation characters in the result, then the string
        is garbage
        Customisation
        =============
        stripping off the last TWO characters as false positives
        included those ending with ').' and similar.
        :param string: string to be tested
        :returns: either True or False
        """

        non_alnum_string = ''.join(char for char in string[1:-2]
                                   if not char.isalnum())
        for char in non_alnum_string[1:]:
            if char != non_alnum_string[0]:
                return True
        return False

    '''
    def has_uppercase_within_lowercase(self, string):
        """
        Rule C
        ======
        If a string begins and ends with a lowercase letter, then if
        the string contains an uppercase letter anywhere in between, then it
        is removed as garbage.
        Customisation
        =============
        false positive on "needed.The". Exclude fullstop-capital.
        Extra customisation: Exclude hyphen-capital, apostrophe-capital and
        forwardslash-capital
        :param string: string to be tested
        :returns: either True or False
        """
        if (string and string[0].islower() and string[-1].islower()):
            string_middle = string[1:-1]
            for index, char in enumerate(string_middle):
                if char.isupper() and not \
                   (index > 0 and string_middle[index-1] in ".-'"):
                    return True
        return False
    '''
    '''
    def googleResults(self, string):

        #If a string has less than 15 Google results, it is garbage.


        address = 'http://www.google.com/search?q=' + string
        results = 0
        page = requests.get(address)
        soup = BeautifulSoup(page.content, 'html.parser')
        resultsString = soup.find(id="resultStats")

        if resultsString is not None:
            results = int(resultsString.text.split(' ')[1].replace(',',''))

        if results < 15:
            return True

        return False
        '''

    def is_garbage(self, string):
        """
        passes the string to check for every rule and if any match is found
        , it returns that rule
        :param string: string to be tested
        :returns: either True or False
        """

        if self.too_long(string):
            return 'L'
        elif self.bad_alnum_ratio(string):
            return 'A'
        elif self.consecutive_four_identical(string):
            return 'R'
        elif self.bad_consonant_vowel_ratio(string):
            return 'V'
        elif self.has_two_distinct_puncts_inside(string):
            return 'P'
        #elif self.has_uppercase_within_lowercase(string):
            #return 'C'
        #elif self.googleResults(string):
            #return 'G'
        return False

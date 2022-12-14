from __future__ import division, print_function

import unicodedata
from functools import partial, lru_cache
from typing import Sequence, Tuple

import numpy as np
from multimethod import multimethod
from uniseg.graphemecluster import grapheme_clusters
#from tqdm import tqdm
from rapidfuzz.distance import Levenshtein

from .extracted_text import ExtractedText
from .config import Config


@multimethod
def distance(s1: str, s2: str):
    """Compute the Levenshtein edit distance between two Unicode strings

    Note that this is different from levenshtein() as this function knows about Unicode
    normalization and grapheme clusters. This should be the correct way to compare two
    Unicode strings.
    """
    seq1 = list(grapheme_clusters(unicodedata.normalize("NFC", s1)))
    seq2 = list(grapheme_clusters(unicodedata.normalize("NFC", s2)))
    return Levenshtein.distance(seq1, seq2)


@multimethod
def distance(s1: ExtractedText, s2: ExtractedText):
    return distance(s1.text, s2.text)


def editops(word1, word2):
    """
    Return sequence of edit operations transforming one string to another.

    Note that this returns indices to the _grapheme clusters_, not characters!
    """
    word1 = list(grapheme_clusters(unicodedata.normalize("NFC", word1)))
    word2 = list(grapheme_clusters(unicodedata.normalize("NFC", word2)))
    return Levenshtein.editops(word1, word2).as_list()
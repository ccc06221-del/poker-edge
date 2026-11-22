"""Cards, two-card combos and the 169 starting-hand classes, as small integers.

A card is rank * 4 + suit, rank 0..12 for 2..A and suit 0..3 for c d h s. A combo
is an unordered pair of cards, numbered 0..1325. A starting-hand class is a cell of
the usual 13 x 13 grid: a pair sits on the diagonal (r * 13 + r), a suited hand
above it (hi * 13 + lo) and an offsuit hand below it (lo * 13 + hi). A flop is an
unordered triple of cards, numbered 0..22099 in the combinatorial number system.
"""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
N_COMBOS = 1326
N_CLASSES = 169
N_FLOPS = 22100


def card(code: str) -> int:
    """'Ah' -> 50."""
    return RANKS.index(code[0].upper()) * 4 + SUITS.index(code[1].lower())


def code(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


# Every combo as two card arrays, lower card first.
COMBO_CARDS = np.array(list(combinations(range(52), 2)), dtype=np.int16)
_COMBO_INDEX = {(int(a), int(b)): i for i, (a, b) in enumerate(COMBO_CARDS)}


def combo_index(c1: int, c2: int) -> int:
    a, b = (c1, c2) if c1 < c2 else (c2, c1)
    return _COMBO_INDEX[(a, b)]


def hand_class(c1: int, c2: int) -> int:
    r1, r2 = c1 // 4, c2 // 4
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return hi * 13 + hi
    return hi * 13 + lo if c1 % 4 == c2 % 4 else lo * 13 + hi


def class_name(k: int) -> str:
    a, b = divmod(k, 13)
    if a == b:
        return RANKS[a] * 2
    return RANKS[a] + RANKS[b] + "s" if a > b else RANKS[b] + RANKS[a] + "o"


COMBO_CLASS = np.array([hand_class(int(a), int(b)) for a, b in COMBO_CARDS], dtype=np.int16)
CLASS_COMBOS = np.bincount(COMBO_CLASS, minlength=N_CLASSES)          # 6, 4 or 12


def flop_index(c1: int, c2: int, c3: int) -> int:
    a, b, c = sorted((c1, c2, c3))
    return comb(a, 1) + comb(b, 2) + comb(c, 3)


FLOP_CARDS = np.zeros((N_FLOPS, 3), dtype=np.int16)
for _a, _b, _c in combinations(range(52), 3):
    FLOP_CARDS[flop_index(_a, _b, _c)] = (_a, _b, _c)

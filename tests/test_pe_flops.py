"""Known-answer tests for cards, the flop taxonomy and the strength buckets."""
import itertools

import numpy as np
import pytest

from pokeredge.cards import (CLASS_COMBOS, COMBO_CARDS, FLOP_CARDS, N_FLOPS, card, class_name, combo_index,
                             flop_index, hand_class)
from pokeredge.flops import (BUCKETS, CLASS_ATTRS, CLASS_KEYS, CLASS_SHARE, FLOP_CLASS, N_CLASS, classify,
                             combo_strength, strength_tensor)


def f(*codes):
    return tuple(card(c) for c in codes)


def s(board, hole):
    b = f(*board.split())
    h1, h2 = f(*hole.split())
    return BUCKETS[combo_strength(b, h1, h2)]


def test_cards_and_classes():
    assert card("2c") == 0 and card("As") == 51
    assert COMBO_CARDS.shape == (1326, 2) and combo_index(card("As"), card("Ah")) == combo_index(card("Ah"), card("As"))
    assert class_name(hand_class(card("Ah"), card("Kh"))) == "AKs"
    assert class_name(hand_class(card("Ah"), card("Kd"))) == "AKo"
    assert class_name(hand_class(card("7h"), card("7d"))) == "77"
    assert sorted(set(CLASS_COMBOS.tolist())) == [4, 6, 12] and CLASS_COMBOS.sum() == 1326


def test_flop_index_is_a_bijection():
    assert FLOP_CARDS.shape == (N_FLOPS, 3)
    assert flop_index(*FLOP_CARDS[12345]) == 12345
    assert flop_index(card("Ah"), card("7d"), card("2c")) == flop_index(card("2c"), card("Ah"), card("7d"))


def test_taxonomy_has_98_classes_and_covers_every_flop():
    assert N_CLASS == 98 and len(CLASS_ATTRS) == 98
    assert np.bincount(FLOP_CLASS, minlength=N_CLASS).min() > 0
    assert CLASS_SHARE.sum() == pytest.approx(1.0)
    kinds = [k[0] for k in CLASS_KEYS]
    assert (kinds.count("u"), kinds.count("mono"), kinds.count("p"), kinds.count("t")) == (63, 8, 26, 1)


def test_known_classes():
    a = CLASS_ATTRS
    k = classify(f("Ah", "7d", "2c"))
    assert k == ("u", 3, "A", "semi", "2nd 9-")             # the ace plays low with the deuce
    assert classify(f("9s", "8s", "7s")) == ("mono", "9-", "conn")
    assert classify(f("Kd", "Kc", "4d")) == ("p", 2, "AA-KK", "odd below", "far")
    assert classify(f("Kd", "7c", "2h")) == ("u", 3, "K", "dis", "2nd 9-")
    assert classify(f("Jh", "Th", "4c")) == ("u", 2, "J", "semi", "2nd T+", "fd top")
    assert classify(f("Jc", "Th", "4h")) == ("u", 2, "J", "semi", "2nd T+", "fd low")
    assert classify(f("5c", "5d", "5h")) == ("t",)
    i = CLASS_KEYS.index(("mono", "9-", "conn"))
    assert a[i]["suits"] == "monotone" and a[i]["wet"] == "wet"
    j = CLASS_KEYS.index(("u", 3, "K", "dis", "2nd 9-"))
    assert a[j]["wet"] == "dry" and a[j]["high4"] == "K-Q" and a[j]["conn"] == "disconnected"


def test_class_is_the_same_under_every_suit_relabelling():
    rng = np.random.default_rng(1)
    for fi in rng.choice(N_FLOPS, 300, replace=False):
        cs = [int(c) for c in FLOP_CARDS[fi]]
        for perm in itertools.permutations(range(4)):
            moved = tuple(c // 4 * 4 + perm[c % 4] for c in cs)
            assert FLOP_CLASS[flop_index(*moved)] == FLOP_CLASS[fi]


def test_suit_isomorphism_gives_1755_flops_each_inside_one_class():
    from pokeredge.flops import iso_flops
    iso = iso_flops()
    assert len(np.unique(iso)) == 1755
    for k in np.unique(iso)[::50]:
        assert len(np.unique(FLOP_CLASS[iso == k])) == 1


def test_strength_buckets_known_answers():
    assert s("Ah 7d 2c", "Ac Kd") == "strong"                # top pair, king kicker
    assert s("Ah 7d 2c", "As 5d") == "medium"                # top pair, weak kicker
    assert s("Ah 7d 2c", "7c 8c") == "medium"                # second pair
    assert s("Ah 7d 2c", "2h 9s") == "weak"                  # bottom pair
    assert s("Ah 7d 2c", "Kc Kd") == "medium"                # pocket pair between A and 7
    assert s("Ah 7d 2c", "5c 5d") == "weak"                  # pocket pair below the second card
    assert s("Kh 7d 2c", "Ac Ad") == "strong"                # overpair
    assert s("Ah 7d 2c", "7c 7h") == "nut"                   # set
    assert s("Ah 7d 2c", "Ac 7c") == "nut"                   # two pair with both cards
    assert s("9h 8d 7c", "Tc 6d") == "nut"                   # straight
    assert s("9h 8h 2c", "Ah 4h") == "draw"                  # nut flush draw, no pair
    assert s("9h 8d 2c", "Tc Jd") == "draw"                  # open-ended
    assert s("9h 8d 2c", "Tc Qd") == "air"                   # gutshot only
    assert s("9h 8h 2c", "2h 4h") == "draw"                  # bottom pair plus a flush draw
    assert s("9h 8h 2c", "Ah Kd") == "air"                   # overcards, backdoor only
    assert s("9h 8h 3h", "Ah 2c") == "draw"                  # one card to the nut flush on a monotone board
    assert s("9h 8h 3h", "Ah 2h") == "nut"                   # flush
    assert s("Kd Kc 7h", "7c 8d") == "medium"                # the odd card below the pair
    assert s("Kd Kc 7h", "Kh 2d") == "nut"                   # trips
    assert s("7d 7c Kh", "Kc 9d") == "medium"                # top pair, 9 kicker
    assert s("7d 7c Kh", "Kc Jd") == "strong"
    assert s("7d 7c Kh", "Qc Qd") == "medium"                # pocket pair between K and 7
    assert s("5c 5d 5h", "Kc Qd") == "medium"
    assert s("5c 5d 5h", "8c 9d") == "air"
    assert s("5c 5d 5h", "2c 2d") == "nut"


def test_strength_tensor_counts_every_unblocked_combo():
    m, c = strength_tensor()
    assert m.shape == (N_FLOPS, 169, 6)
    assert (m.sum(axis=(1, 2)) == 1176).all()               # 49 choose 2 combos left after any flop
    fi = flop_index(*f("Ah", "7d", "2c"))
    aa = hand_class(card("Ac"), card("Ad"))
    assert c[fi, aa] == 3 and m[fi, aa, BUCKETS.index("nut")] == 3

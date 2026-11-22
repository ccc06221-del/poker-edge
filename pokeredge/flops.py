"""The flop taxonomy (98 classes) and hand-strength buckets on the flop.

Taxonomy
--------
Kieran's brief names four dimensions: suit pattern, high card, connectedness and
pairing. Crossed in full at 3 x 4 x 3 x 3 they give 108 cells, but only 55 can
occur (a monotone board cannot be paired, a paired board cannot hold a straight,
an unpaired A-high board is never disconnected, ...). So the classes are a tree
on those four dimensions, plus two splits that matter for ranges, chosen so every
class can occur:

- Unpaired, rainbow or two-tone (63 classes): high card (A, K, Q, J, T, 9 or lower)
  x connectedness x second card (T or higher, or not); two-tone boards are also
  split by whether the top card is one of the two suited cards (who holds the nut
  flush draw).
- Unpaired, monotone (8): high card (A, K-Q, J-T, 9 or lower) x straight possible.
- Paired (26): rainbow or two-tone x rank of the pair (A-K, Q-T, 9-7, 6-2) x the
  odd card above or below the pair x the two ranks within four of each other or not.
- Trips (1).

Connectedness on an unpaired board: *connected* if two hole cards can make a
straight (the three ranks fit in five, ace low or high); *semi* if not, but two of
the ranks are within four of each other (straight draws exist); *disconnected*
otherwise. The class of a flop depends only on its ranks and suit pattern, so it is
the same for every suit relabelling (tested).

Every class also carries the four brief dimensions (`pairing`, `suits`, `high4`,
`conn`) and a wetness level, which is what the coarser signals group on.

Strength buckets
----------------
Six, judged on the flop alone (two hole cards and three board cards):

    nut     two pair using both hole cards, a set, trips, a straight, a flush or better
    strong  an overpair, or top pair with a kicker of T or better
    medium  top pair with a weaker kicker, second pair, or a pocket pair between the
            top two board ranks
    draw    no pair better than weak, with a flush draw or an open-ended or
            double-gutshot straight draw (two or more ranks complete a straight)
    weak    bottom pair, or a pocket pair below the second board rank
    air     everything else (overcards, gutshots, ace high)

On a trips board: a pocket pair or a matching card is `nut`, a queen or better is
`medium`, anything else `air`.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations

import numpy as np

from .cards import COMBO_CARDS, COMBO_CLASS, FLOP_CARDS, N_CLASSES, N_FLOPS

BUCKETS = ("air", "weak", "draw", "medium", "strong", "nut")
AIR, WEAK, DRAW, MEDIUM, STRONG, NUT = range(6)
K = len(BUCKETS)

T_RANK = 8          # rank index of the ten


def _straight_tables() -> tuple[np.ndarray, np.ndarray]:
    """For every 13-bit rank mask: does it hold a straight, and how many missing ranks would make one."""
    windows = [0b1111 << 0 | 1 << 12] + [0b11111 << k for k in range(9)]   # wheel A2345, then 23456 .. TJQKA
    masks = np.arange(1 << 13)
    has = np.zeros(1 << 13, dtype=bool)
    for w in windows:
        has |= (masks & w) == w
    outs = np.zeros(1 << 13, dtype=np.int8)
    for r in range(13):
        bit = 1 << r
        outs += ((masks & bit) == 0) & has[masks | bit]
    return has, outs


HAS_STRAIGHT, STRAIGHT_OUTS = _straight_tables()


# ---------------------------------------------------------------- taxonomy

def _straight_possible(ranks: list[int]) -> bool:
    d = sorted(set(ranks))
    if len(d) < 3:
        return False
    alts = [d] + ([sorted(-1 if x == 12 else x for x in d)] if 12 in d else [])
    return any(a[-1] - a[0] <= 4 for a in alts)


def _min_gap(ranks: list[int]) -> int:
    d = sorted(set(ranks))
    best = 99
    for a, b in combinations(d, 2):
        g = b - a
        if b == 12:
            g = min(g, a + 1)          # the ace plays low too
        best = min(best, g)
    return best


def _hi6(r: int) -> str:
    return {12: "A", 11: "K", 10: "Q", 9: "J", 8: "T"}.get(r, "9-")


def _hi4(r: int) -> str:
    return "A" if r == 12 else "K-Q" if r >= 10 else "J-T" if r >= 8 else "9-"


def classify(cards: tuple[int, int, int]) -> tuple:
    """The class key of one flop (cards as integers)."""
    cs = sorted(cards, key=lambda c: (c // 4, c % 4), reverse=True)
    rs = [c // 4 for c in cs]
    suits = len({c % 4 for c in cs})
    nd = len(set(rs))
    if nd == 3:
        if suits == 1:
            return ("mono", _hi4(rs[0]), "conn" if _straight_possible(rs) else "not")
        conn = "conn" if _straight_possible(rs) else "semi" if _min_gap(rs) <= 4 else "dis"
        key = ("u", suits, _hi6(rs[0]), conn, "2nd T+" if rs[1] >= T_RANK else "2nd 9-")
        if suits == 2:
            key += ("fd top" if cs[0] % 4 in (cs[1] % 4, cs[2] % 4) else "fd low",)
        return key
    if nd == 2:
        pr = rs[1]                                   # sorted x x y or y x x: the middle card is paired
        pb = "AA-KK" if pr >= 11 else "QQ-TT" if pr >= 8 else "99-77" if pr >= 5 else "66-22"
        odd = "odd above" if rs[0] != rs[1] else "odd below"
        return ("p", suits, pb, odd, "near" if _min_gap(rs) <= 4 else "far")
    return ("t",)


def _attrs(key: tuple) -> dict:
    kind = key[0]
    if kind == "u":
        _, suits, hi6, conn, mid, *fd = key
        suit = {3: "rainbow", 2: "two-tone"}[suits]
        high = "A" if hi6 == "A" else "K-Q" if hi6 in ("K", "Q") else "J-T" if hi6 in ("J", "T") else "9-"
        a = {"pairing": "unpaired", "suits": suit, "high4": high, "conn": {"conn": "connected", "semi": "semi",
             "dis": "disconnected"}[conn]}
        label = f"{hi6}-high {suit} {a['conn']}, {mid}" + (f", {fd[0]}" if fd else "")
    elif kind == "mono":
        _, high, conn = key
        a = {"pairing": "unpaired", "suits": "monotone", "high4": high,
             "conn": "connected" if conn == "conn" else "semi/disconnected"}
        label = f"{high}-high monotone, {'straight possible' if conn == 'conn' else 'no straight'}"
    elif kind == "p":
        _, suits, pb, odd, near = key
        suit = {3: "rainbow", 2: "two-tone"}[suits]
        a = {"pairing": "paired", "suits": suit, "high4": None, "conn": "semi" if near == "near" else "disconnected"}
        label = f"paired {pb} {suit}, {odd}, {near}"
    else:
        a = {"pairing": "trips", "suits": "rainbow", "high4": None, "conn": "disconnected"}
        label = "trips"
    flush = {"rainbow": 0, "two-tone": 1, "monotone": 2}[a["suits"]]
    straight = {"connected": 2, "semi": 1}.get(a["conn"], 0)
    wet = flush + straight - (1 if a["pairing"] != "unpaired" else 0)
    a["wet"] = "wet" if wet >= 2 else "medium" if wet == 1 else "dry"
    a["label"] = label
    return a


def _build():
    keys = [classify(tuple(int(c) for c in f)) for f in FLOP_CARDS]
    order = sorted(set(keys), key=lambda k: (["u", "mono", "p", "t"].index(k[0]), str(k)))
    cid = {k: i for i, k in enumerate(order)}
    flop_class = np.array([cid[k] for k in keys], dtype=np.int16)
    attrs = [_attrs(k) for k in order]
    # high4 for paired and trips boards comes from the board's top rank; filled per class from its flops.
    for i, a in enumerate(attrs):
        if a["high4"] is None:
            f = FLOP_CARDS[np.flatnonzero(flop_class == i)[0]]
            a["high4"] = _hi4(int(max(f) // 4))
    return order, flop_class, attrs


CLASS_KEYS, FLOP_CLASS, CLASS_ATTRS = _build()
N_CLASS = len(CLASS_KEYS)
CLASS_SHARE = np.bincount(FLOP_CLASS, minlength=N_CLASS) / N_FLOPS


@lru_cache(maxsize=1)
def iso_flops() -> np.ndarray:
    """For every flop, the id of its suit-isomorphism class (1,755 strategically distinct flops)."""
    perms = list(permutations(range(4)))
    keys = {}
    out = np.empty(N_FLOPS, dtype=np.int32)
    for f in range(N_FLOPS):
        cs = [int(c) for c in FLOP_CARDS[f]]
        key = min(tuple(sorted((c // 4 * 4 + p[c % 4] for c in cs), reverse=True)) for p in perms)
        out[f] = keys.setdefault(key, len(keys))
    return out


def flop_class_of(cards) -> int:
    from .cards import flop_index
    return int(FLOP_CLASS[flop_index(*cards)])


# ---------------------------------------------------------------- strength

def strength(flop: tuple[int, int, int], c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Strength bucket of each hole-card pair (c1[i], c2[i]) on one flop. Vectorised over pairs."""
    c1 = np.asarray(c1, dtype=np.int64)
    c2 = np.asarray(c2, dtype=np.int64)
    b = np.array(flop, dtype=np.int64)
    br, bs = b // 4, b % 4
    r1, r2 = c1 // 4, c2 // 4
    hi, lo = np.maximum(r1, r2), np.minimum(r1, r2)
    s1, s2 = c1 % 4, c2 % 4
    bcnt = np.bincount(br, minlength=13)
    distinct = sorted(set(br.tolist()), reverse=True)
    nd = len(distinct)
    pocket = r1 == r2
    h_hi, h_lo = bcnt[hi], bcnt[lo]

    full = np.int64(np.bitwise_or.reduce(1 << br)) | (1 << r1) | (1 << r2)
    distinct5 = (~pocket) & (h_hi == 0) & (h_lo == 0) & (nd == 3)
    straight = distinct5 & HAS_STRAIGHT[full]
    suit_n = np.stack([(bs == s).sum() + (s1 == s) + (s2 == s) for s in range(4)], axis=1)
    flush = suit_n.max(axis=1) >= 5
    fd_suit = suit_n.argmax(axis=1)
    flush_draw = (suit_n.max(axis=1) == 4) & ((s1 == fd_suit) | (s2 == fd_suit))
    oesd = STRAIGHT_OUTS[full] >= 2

    out = np.full(len(c1), AIR, dtype=np.int8)
    if nd == 1:                                          # trips on board
        out[hi >= 10] = MEDIUM
        out[pocket | (h_hi > 0) | (h_lo > 0)] = NUT
        return out

    top = distinct[0]
    second = distinct[1]
    nut = straight | flush | (pocket & (h_hi >= 1)) | (~pocket & ((h_hi == 2) | (h_lo == 2)))
    nut |= ~pocket & (h_hi >= 1) & (h_lo >= 1)

    # One pair made with a hole card (or a pocket pair that missed).
    one_hit = ~pocket & ((h_hi >= 1) ^ (h_lo >= 1))
    hit_rank = np.where(h_hi >= 1, hi, lo)
    kicker = np.where(h_hi >= 1, lo, hi)
    top_once = bcnt[top] == 1
    top_pair = one_hit & (hit_rank == top) & top_once
    made = np.full(len(c1), -1, dtype=np.int8)
    made[top_pair & (kicker >= T_RANK)] = STRONG
    made[top_pair & (kicker < T_RANK)] = MEDIUM
    if nd == 3:
        made[one_hit & (hit_rank == second)] = MEDIUM
        made[one_hit & (hit_rank == distinct[2])] = WEAK
    else:                                                # paired board: the odd card is top or second
        made[one_hit & (hit_rank == second) & (bcnt[second] == 1)] = MEDIUM
    miss_pp = pocket & (h_hi == 0)
    made[miss_pp & (hi > top)] = STRONG
    made[miss_pp & (hi < top) & (hi > second)] = MEDIUM
    made[miss_pp & (hi < second)] = WEAK

    out[made == WEAK] = WEAK
    out[((made == WEAK) | (made == -1)) & (flush_draw | oesd)] = DRAW
    out[made == MEDIUM] = MEDIUM
    out[made == STRONG] = STRONG
    out[nut] = NUT
    return out


def strength_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The bucket of every combo on every flop, and the counts the range model needs.

    B[flop, combo]: bucket, or -1 where the flop blocks the combo.
    M[flop, class, bucket]: combos of each starting-hand class in each bucket, after card removal.
    C[flop, class]: combos of the class that the flop does not block.
    """
    bt = np.full((N_FLOPS, len(COMBO_CARDS)), -1, dtype=np.int8)
    m = np.zeros((N_FLOPS, N_CLASSES, K), dtype=np.uint8)
    a, b = COMBO_CARDS[:, 0].astype(np.int64), COMBO_CARDS[:, 1].astype(np.int64)
    for f in range(N_FLOPS):
        fc = FLOP_CARDS[f]
        ok = ~(np.isin(a, fc) | np.isin(b, fc))
        s = strength(tuple(int(x) for x in fc), a[ok], b[ok])
        bt[f, ok] = s
        np.add.at(m[f], (COMBO_CLASS[ok], s), 1)
    return bt, m, m.sum(axis=2).astype(np.uint8)


def strength_tensor() -> tuple[np.ndarray, np.ndarray]:
    """M and C from `strength_tables`."""
    _, m, c = strength_tables()
    return m, c


def combo_strength(flop: tuple[int, int, int], c1: int, c2: int) -> int:
    return int(strength(flop, np.array([c1]), np.array([c2]))[0])

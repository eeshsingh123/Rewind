import math
from collections.abc import Sequence
from dataclasses import dataclass

from rewind.core.analysis.timeline import Judged, Stop

# A pattern is notable when chance alone would make it this rarely, after counting every place
# we looked; failures must also repeat, so one bad item is never a pattern.
NOTABLE_P = 0.05
MIN_FAILURES = 3
# "Every Nth item" is looked for up to this N, and only where there are 3 full cycles.
MAX_PERIOD = 12
# Items this many usual gaps after a restart count as "right after" it.
RESTART_ITEMS = 3


@dataclass(frozen=True, slots=True)
class Pattern:
    kind: str
    # Chance of something at least this strong if failures fell at random.
    p: float
    # The failed items the pattern is about, in time order.
    items: tuple[Judged, ...]
    # in_a_row: (run,); every_nth: (period, place of each item in order...);
    # after_restart and started/stopped: (failed, of, failed elsewhere, of elsewhere).
    numbers: tuple[int, ...]
    # started/stopped: when the change happened.
    at: float | None = None

    @property
    def notable(self) -> bool:
        return self.p < NOTABLE_P


@dataclass(frozen=True, slots=True)
class Split:
    before: tuple[Judged, ...]
    after: tuple[Judged, ...]
    p: float


def patterns(judged: Sequence[Judged], stops: Sequence[Stop]) -> tuple[Pattern, ...]:
    ordered = sorted(judged, key=lambda j: j.t)
    failed = [j for j in ordered if not j.passed]
    if len(failed) < 2 or len(failed) == len(ordered):
        return ()
    found = [_run(ordered), _period(ordered), _after_restart(ordered, stops)]
    split = best_split(ordered)
    if split is not None:
        found.append(_shift(split))
    return tuple(p for p in found if p is not None)


def best_split(ordered: Sequence[Judged]) -> Split | None:
    n, k = len(ordered), sum(not j.passed for j in ordered)
    best: Split | None = None
    for s in range(MIN_FAILURES, n - MIN_FAILURES + 1):
        before, after = tuple(ordered[:s]), tuple(ordered[s:])
        after_failed = sum(not j.passed for j in after)
        p = min(
            _fisher_more(after_failed, len(after), k, n),
            _fisher_more(k - after_failed, len(before), k, n),
        )
        if best is None or p < best.p:
            best = Split(before, after, p)
    if best is None:
        return None
    # Every split point was a chance to find one.
    return Split(best.before, best.after, min(1.0, best.p * (n - 2 * MIN_FAILURES + 1)))


def compare(this: Sequence[Judged], earlier: Sequence[Judged]) -> float:
    """Chance of a difference at least this large between two sets of items, in its direction."""
    n, k = len(this) + len(earlier), sum(not j.passed for j in (*this, *earlier))
    this_failed = sum(not j.passed for j in this)
    if not n or not k:
        return 1.0
    expected = k * len(this) / n
    if this_failed >= expected:
        return _fisher_more(this_failed, len(this), k, n)
    return _fisher_more(k - this_failed, len(earlier), k, n)


def _run(ordered: Sequence[Judged]) -> Pattern | None:
    longest: list[Judged] = []
    current: list[Judged] = []
    for item in ordered:
        current = [*current, item] if not item.passed else []
        if len(current) > len(longest):
            longest = current
    if len(longest) < MIN_FAILURES:
        return None
    n, k = len(ordered), sum(not j.passed for j in ordered)
    p = 1 - _arrangements_without_run(n, k, len(longest)) / math.comb(n, k)
    return Pattern("in_a_row", p, tuple(longest), (len(longest),))


def _period(ordered: Sequence[Judged]) -> Pattern | None:
    n, k = len(ordered), sum(not j.passed for j in ordered)
    best: Pattern | None = None
    tests = 0
    for period in range(2, min(MAX_PERIOD, n // 3) + 1):
        for phase in range(period):
            tests += 1
            members = ordered[phase::period]
            hits = tuple(j for j in members if not j.passed)
            # Most of the failures, and most of the members, or it is not one faulty head.
            if len(hits) < MIN_FAILURES or len(hits) < 0.75 * k or len(hits) < 0.5 * len(members):
                continue
            p = _fisher_more(len(hits), len(members), k, n)
            if best is None or p < best.p:
                places = tuple(ordered.index(j) + 1 for j in hits)
                best = Pattern("every_nth", p, hits, (period, *places))
    if best is None:
        return None
    return Pattern(best.kind, min(1.0, best.p * tests), best.items, best.numbers)


def _after_restart(ordered: Sequence[Judged], stops: Sequence[Stop]) -> Pattern | None:
    after: list[Judged] = []
    for stop in stops:
        window = stop.gap.end + RESTART_ITEMS * stop.usual_gap_s
        after += [j for j in ordered if stop.gap.end <= j.t <= window]
    hits = tuple(j for j in after if not j.passed)
    if len(hits) < 2:
        return None
    n, k = len(ordered), sum(not j.passed for j in ordered)
    p = _fisher_more(len(hits), len(after), k, n)
    return Pattern("after_restart", p, hits, (len(hits), len(after), k - len(hits), n - len(after)))


def _shift(split: Split) -> Pattern:
    before_failed = sum(not j.passed for j in split.before)
    after_failed = sum(not j.passed for j in split.after)
    started = after_failed / len(split.after) > before_failed / len(split.before)
    side = split.after if started else split.before
    numbers = (before_failed, len(split.before), after_failed, len(split.after))
    return Pattern("started" if started else "stopped", split.p,
                   tuple(j for j in side if not j.passed), numbers, split.after[0].t)  # fmt: skip


# Chance that a group of `size` drawn from n items, k of them failed, holds `hits` or more failures.
def _fisher_more(hits: int, size: int, k: int, n: int) -> float:
    total = math.comb(n, size)
    upper = min(size, k)
    return sum(math.comb(k, i) * math.comb(n - k, size - i) for i in range(hits, upper + 1)) / total


# Ways to place k failures among n items with no run of `run` or more in a row.
def _arrangements_without_run(n: int, k: int, run: int) -> int:
    slots = n - k + 1
    total = 0
    for j in range(slots + 1):
        rest = k - j * run
        if rest < 0:
            break
        total += (-1) ** j * math.comb(slots, j) * math.comb(rest + slots - 1, slots - 1)
    return total

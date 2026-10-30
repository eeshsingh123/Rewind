import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from rewind.contracts import TimeSpan
from rewind.core.ask.marks import Mark

# "measured": code counted or timed it from tracks. "seen": Muse judged it from video or crops.
type Kind = Literal["measured", "seen"]

MIN_WORDS = 3
MAX_CLAIM_CHARS = 500
_EVIDENCE_ID = re.compile(r"\be\d+\b")
_NUMBER = re.compile(r"\d+(?:[:.]\d+)*")
_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
}  # fmt: skip
_WORD = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    kind: Kind
    # States every number and time a claim citing it may use, times as mm:ss.
    text: str
    # The stretch a proof shows, and the boxes drawn on it. A photo has no stretch: any span
    # means the photo itself.
    clip: TimeSpan | None = None
    marks: tuple[Mark, ...] = ()
    # A proof already made, for evidence no single stretch can show.
    proof_key: str | None = None

    @property
    def shows(self) -> bool:
        return self.clip is not None or self.proof_key is not None

    @property
    def line(self) -> str:
        return f"{self.id} ({self.kind}): {self.text}"


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    # Measured only when every piece of evidence behind it was measured.
    kind: Kind
    evidence: tuple[Evidence, ...]

    # The first cited evidence that can be shown, with every other cited item in its stretch
    # boxed too: a claim about two boxes shows two.
    @property
    def shown(self) -> Evidence | None:
        first = next((e for e in self.evidence if e.shows), None)
        if first is None or first.clip is None:
            return first
        marks = list(first.marks)
        for other in self.evidence:
            if other.clip is not None and _overlap(other.clip, first.clip):
                marks += [mark for mark in other.marks if mark not in marks]
        return replace(first, marks=tuple(marks))


# Every tool result is written down here with an id, so an answer can only cite what was found.
class Notebook:
    def __init__(self) -> None:
        self._entries: dict[str, Evidence] = {}

    def add(
        self,
        kind: Kind,
        text: str,
        clip: TimeSpan | None = None,
        marks: Sequence[Mark] = (),
        proof_key: str | None = None,
    ) -> Evidence:
        number = len(self._entries) + 1
        evidence = Evidence(f"e{number}", kind, text, clip, tuple(marks), proof_key)
        self._entries[evidence.id] = evidence
        return evidence

    # What code worked out itself: it stands as the answer when the agent's own words do not.
    def worked_out(self) -> tuple[Claim, ...]:
        kept = [e for e in self._entries.values() if e.proof_key is not None]
        return tuple(Claim(e.text, e.kind, (e,)) for e in kept)

    # A claim is kept only when it cites known evidence and states no number or time beyond it.
    def check(self, text: str, cited: Sequence[str]) -> Claim | str:
        text = " ".join(text.split())
        # Word counts only mean something in scripts that put spaces between words.
        short = text.isascii() and len(text.split()) < MIN_WORDS
        if short or len(text) > MAX_CLAIM_CHARS:
            return f"Write a plain sentence of at most {MAX_CLAIM_CHARS} characters."
        if _EVIDENCE_ID.search(text):
            return "Do not write evidence ids in the sentence."
        unknown = [i for i in cited if i not in self._entries]
        backing = tuple(self._entries[i] for i in dict.fromkeys(cited) if i in self._entries)
        if unknown or not backing:
            return f"Cite evidence ids from the list (unknown: {', '.join(unknown) or 'none'})."
        # The number of things cited is a count code can vouch for.
        missing = untraceable(text, [*(e.text for e in backing), str(len(backing))])
        if missing:
            return (
                f"These numbers or times are not in the evidence cited: {', '.join(missing)}. "
                "Use them only as written there."
            )
        kind: Kind = "measured" if all(e.kind == "measured" for e in backing) else "seen"
        if kind == "seen" and not any(e.shows for e in backing):
            return "Nothing cited can be shown. Look at the items or frames, then cite those."
        return Claim(text, kind, backing)


# Numbers and times a sentence states that its sources do not: a claim code cannot trace.
def untraceable(text: str, sources: Sequence[str]) -> tuple[str, ...]:
    known = {_canonical(n) for source in sources for n in _NUMBER.findall(source)}
    spoken = _WORD.sub(lambda m: _WORDS[m.group(1).lower()], text)
    stated = [n for n in _NUMBER.findall(spoken) if _canonical(n) not in {"0", "1"}]
    return tuple(dict.fromkeys(n for n in stated if _canonical(n) not in known))


def _overlap(one: TimeSpan, other: TimeSpan) -> bool:
    return one.start <= other.end and other.start <= one.end


def _canonical(number: str) -> str:
    if ":" in number:
        seconds = 0
        for part in number.split(":"):
            if not part.isdigit():
                return number
            seconds = seconds * 60 + int(part)
        return f"t{seconds}"
    try:
        return f"{float(number):g}"
    except ValueError:
        return number

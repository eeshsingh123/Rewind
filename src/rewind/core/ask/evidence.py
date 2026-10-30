import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rewind.contracts import TimeSpan
from rewind.core.analysis.facts import Fact, untraceable
from rewind.core.analysis.proof import Mark

# "measured": code counted or timed it from tracks. "seen": Muse judged it from video or crops.
type Kind = Literal["measured", "seen"]

MIN_WORDS = 3
MAX_CLAIM_CHARS = 300
_EVIDENCE_ID = re.compile(r"\be\d+\b")


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

    @property
    def line(self) -> str:
        return f"{self.id} ({self.kind}): {self.text}"


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    # Measured only when every piece of evidence behind it was measured.
    kind: Kind
    evidence: tuple[Evidence, ...]

    # The first cited evidence with a clip: what the claim's proof shows.
    @property
    def shown(self) -> Evidence | None:
        return next((e for e in self.evidence if e.clip is not None), None)


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
    ) -> Evidence:
        evidence = Evidence(f"e{len(self._entries) + 1}", kind, text, clip, tuple(marks))
        self._entries[evidence.id] = evidence
        return evidence

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
        missing = untraceable(text, [Fact(e.id, e.text) for e in backing])
        if missing:
            return (
                f"These numbers or times are not in the evidence cited: {', '.join(missing)}. "
                "Use them only as written there."
            )
        kind: Kind = "measured" if all(e.kind == "measured" for e in backing) else "seen"
        return Claim(text, kind, backing)

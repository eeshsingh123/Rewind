from collections.abc import Sequence


def clock_time(t: float) -> str:
    minutes, seconds = divmod(int(t), 60)
    return f"{minutes:02d}:{seconds:02d}"


def plural(phrase: str, count: int) -> str:
    if count == 1:
        return phrase
    if phrase.endswith(("s", "x", "z", "ch", "sh")):
        return phrase + "es"
    if phrase.endswith("y") and phrase[-2:-1] not in ("a", "e", "i", "o", "u"):
        return phrase[:-1] + "ies"
    return phrase + "s"


def join_words(words: Sequence[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]

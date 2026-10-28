from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from rewind.contracts import Footage, RuleSpec, SamplingPlan, TimeSpan


@dataclass(frozen=True, slots=True)
class PlanInput:
    footage: Footage
    active_spans: tuple[TimeSpan, ...]
    rules: tuple[RuleSpec, ...]
    fps: float


class SamplingStrategy(Protocol):
    def plan(self, request: PlanInput) -> SamplingPlan: ...


class UnknownStrategyError(Exception):
    pass


class Planner:
    def __init__(self, strategies: Mapping[str, SamplingStrategy]) -> None:
        self._strategies = strategies

    def plan(self, strategy: str, request: PlanInput) -> SamplingPlan:
        chosen = self._strategies.get(strategy)
        if chosen is None:
            known = ", ".join(sorted(self._strategies))
            raise UnknownStrategyError(f"unknown sampling strategy {strategy!r}; known: {known}")
        return chosen.plan(request)


def rule_concepts(rules: Sequence[RuleSpec]) -> tuple[str, ...]:
    phrases = [phrase for rule in rules for phrase in (rule.entity.phrase, *rule.parts.values())]
    return tuple(dict.fromkeys(phrases))

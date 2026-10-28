from decimal import Decimal

from pydantic import AwareDatetime, Field

from rewind.contracts.base import Contract, Id, Key


class Charge(Contract):
    provider: Key
    model: Id
    # What the provider bills by, e.g. video_frame, image, input_token, cached_input_token.
    unit: Key
    quantity: int = Field(ge=0)
    usd: Decimal = Field(ge=0)


class CostEntry(Charge):
    id: Id
    at: AwareDatetime
    workspace_id: Id
    job_id: Id | None = None


class CostEstimate(Contract):
    charges: tuple[Charge, ...]

    @property
    def total_usd(self) -> Decimal:
        return sum((charge.usd for charge in self.charges), Decimal(0))

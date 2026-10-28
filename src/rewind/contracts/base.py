from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Id = Annotated[str, StringConstraints(min_length=1)]
Key = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
Concept = Annotated[str, StringConstraints(min_length=1)]
Seconds = Annotated[float, Field(ge=0)]


class TimeSpan(Contract):
    start: Seconds
    end: Seconds

    @model_validator(mode="after")
    def _end_not_before_start(self) -> Self:
        if self.end < self.start:
            raise ValueError("end must not be before start")
        return self

"""Validated public contracts for the controlled EditGuard support domain."""

from datetime import date
from decimal import Decimal, InvalidOperation
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Relation = Literal["device_limit", "price_monthly", "storage_gb", "support_hours"]


def canonical_numeric(value: str) -> str:
    """Restrict initial facts to nonnegative, finite plain decimal values."""
    if not isinstance(value, str) or not re.fullmatch(r"\d+(?:\.\d+)?", value.strip()):
        raise ValueError("value must be a nonnegative plain decimal string")
    try:
        number = Decimal(value.strip())
    except InvalidOperation as exc:
        raise ValueError("invalid numeric value") from exc
    return format(number.normalize(), "f")


def iso_date(value):
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError("date must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError("date must use YYYY-MM-DD")
    return parsed


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Fact(StrictModel):
    fact_id: str = Field(min_length=1, max_length=120)
    entity_id: str = Field(min_length=1, max_length=120)
    entity_name: str = Field(min_length=1, max_length=120)
    aliases: list[str] = Field(default_factory=list)
    relation: Relation
    value: str = Field(max_length=40)
    unit: str = Field(min_length=1, max_length=80)
    plan: str = Field(min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=80)
    effective_from: date
    effective_to: date | None = None
    source: str = Field(min_length=1, max_length=1000)
    lineage_id: str = Field(min_length=1, max_length=120)

    _dates = field_validator("effective_from", "effective_to", mode="before")(
        lambda value: None if value is None else iso_date(value)
    )
    _value = field_validator("value", mode="before")(canonical_numeric)

    @field_validator("aliases")
    @classmethod
    def aliases_valid(cls, values):
        if any(not alias.strip() or len(alias) > 120 for alias in values):
            raise ValueError("aliases must be nonempty and at most 120 characters")
        return list(dict.fromkeys(alias.strip() for alias in values))

    @model_validator(mode="after")
    def interval_valid(self):
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be after effective_from (exclusive)")
        if self.relation != "price_monthly" and Decimal(self.value) != Decimal(self.value).to_integral_value():
            raise ValueError("limits and support hours must be whole numbers")
        if self.relation == "support_hours" and Decimal(self.value) > 24:
            raise ValueError("support_hours must be between 0 and 24 hours per day")
        return self


class CorrectionCreate(StrictModel):
    fact_id: str = Field(min_length=1, max_length=120)
    original_value: str = Field(max_length=40)
    corrected_value: str = Field(max_length=40)
    source: str = Field(min_length=1, max_length=1000)
    reason: str = Field(min_length=1, max_length=2000)
    effective_from: date
    effective_to: date | None = None
    supersedes: str | None = Field(default=None, min_length=1, max_length=120)

    _dates = field_validator("effective_from", "effective_to", mode="before")(
        lambda value: None if value is None else iso_date(value)
    )
    _values = field_validator("original_value", "corrected_value", mode="before")(canonical_numeric)

    @model_validator(mode="after")
    def interval_valid(self):
        if self.effective_to is not None and self.effective_to <= self.effective_from:
            raise ValueError("effective_to must be after effective_from (exclusive)")
        if self.original_value == self.corrected_value:
            raise ValueError("a correction must change the value")
        return self


class AskRequest(StrictModel):
    question: str = Field(min_length=1, max_length=1000)
    entity_id: str | None = Field(default=None, min_length=1, max_length=120)
    plan: str | None = Field(default=None, min_length=1, max_length=80)
    version: str | None = Field(default=None, min_length=1, max_length=80)
    as_of: date | None = None
    mode: Literal["structured", "learned", "unchanged_base", "updated_rag", "prompt_only", "similarity_editor"] = "structured"

    _date = field_validator("as_of", mode="before")(
        lambda value: None if value is None else iso_date(value)
    )

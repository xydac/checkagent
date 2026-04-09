"""Golden dataset schema for eval cases.

Defines the EvalCase model used by golden datasets for parametrized
evaluation of agent runs.

Requirements: F3.2
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class EvalCase(BaseModel):
    """A single test case in a golden dataset.

    Example YAML/JSON:
        id: refund-001
        input: "I want to return order #12345"
        expected_tools: [lookup_order, check_return_policy, initiate_refund]
        expected_output_contains: ["refund initiated", "3-5 business days"]
        max_steps: 8
        tags: [refund, happy-path]
    """

    id: str
    input: str
    expected_tools: list[str] = Field(default_factory=list)
    expected_output_contains: list[str] = Field(default_factory=list)
    expected_output_equals: str | None = None
    max_steps: int | None = None
    tags: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_constraints(self) -> EvalCase:
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        return self


class GoldenDataset(BaseModel):
    """A collection of test cases forming a golden dataset.

    Includes optional metadata about the dataset itself (name, version, etc.).
    """

    name: str = "unnamed"
    version: str = "1"
    description: str = ""
    cases: list[EvalCase]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(cls, v: Any) -> str:
        """Accept int/float version values (e.g. YAML ``version: 2``)."""
        if isinstance(v, (int, float)):
            return str(v)
        return v

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> GoldenDataset:
        ids = [c.id for c in self.cases]
        dupes = [i for i in ids if ids.count(i) > 1]
        if dupes:
            raise ValueError(f"Duplicate test case IDs: {set(dupes)}")
        return self

    def filter_by_tags(self, *tags: str) -> list[EvalCase]:
        """Return cases matching any of the given tags."""
        tag_set = set(tags)
        return [c for c in self.cases if tag_set & set(c.tags)]

    def get_case(self, case_id: str) -> EvalCase | None:
        """Look up a single test case by ID."""
        for c in self.cases:
            if c.id == case_id:
                return c
        return None

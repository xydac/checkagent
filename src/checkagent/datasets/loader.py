"""Golden dataset loader.

Loads test cases from JSON and YAML files, validates them against
the EvalCase schema, and provides pytest parametrize integration.

Requirements: F3.2, F3.3
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from checkagent.datasets.schema import EvalCase, GoldenDataset


def _load_raw(path: Path) -> dict[str, Any] | list[Any]:
    """Load raw data from a JSON or YAML file."""
    suffix = path.suffix.lower()

    if suffix in (".json",):
        with open(path) as f:
            return json.load(f)

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML datasets. "
                "Install it with: pip install pyyaml"
            ) from None
        with open(path) as f:
            return yaml.safe_load(f)

    raise ValueError(f"Unsupported file format: {suffix} (expected .json, .yaml, or .yml)")


def _normalize(raw: dict[str, Any] | list[Any]) -> dict[str, Any]:
    """Normalize raw data into the GoldenDataset dict format.

    Accepts either:
    - A list of test case dicts (bare format)
    - A dict with a 'cases' key (full format with optional metadata)
    """
    if isinstance(raw, list):
        return {"cases": raw}

    if isinstance(raw, dict):
        if "cases" in raw:
            return raw
        raise ValueError(
            "Dataset dict must contain a 'cases' key. "
            "Alternatively, provide a bare list of test case objects."
        )

    raise ValueError(f"Expected list or dict, got {type(raw).__name__}")


def load_dataset(path: str | Path) -> GoldenDataset:
    """Load and validate a golden dataset from a file.

    Args:
        path: Path to a JSON or YAML file containing test cases.

    Returns:
        A validated GoldenDataset instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is unsupported or data is invalid.
        ValidationError: If test cases fail schema validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw = _load_raw(path)
    normalized = _normalize(raw)
    return GoldenDataset.model_validate(normalized)


def load_cases(path: str | Path, tags: list[str] | None = None) -> list[EvalCase]:
    """Load test cases from a file, optionally filtered by tags.

    Convenience function that returns just the list of EvalCase objects.

    Args:
        path: Path to a JSON or YAML file.
        tags: If provided, only return cases matching any of these tags.

    Returns:
        List of validated EvalCase instances.
    """
    dataset = load_dataset(path)
    if tags:
        return dataset.filter_by_tags(*tags)
    return dataset.cases


def parametrize_cases(
    path: str | Path, tags: list[str] | None = None
) -> tuple[str, list[Any]]:
    """Generate pytest.mark.parametrize arguments from a golden dataset.

    Usage:
        @pytest.mark.parametrize(*parametrize_cases("golden.json"))
        async def test_agent(test_case, my_agent):
            run = await my_agent.run(test_case.input)
            ...

    Args:
        path: Path to a JSON or YAML golden dataset file.
        tags: If provided, only include cases matching any of these tags.

    Returns:
        A tuple of (argname, argvalues) suitable for pytest.mark.parametrize.
        Each argvalue is an EvalCase with its id set as the pytest ID.
    """
    import pytest

    cases = load_cases(path, tags=tags)
    return (
        "test_case",
        [pytest.param(case, id=case.id) for case in cases],
    )

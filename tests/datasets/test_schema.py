"""Tests for golden dataset schema validation."""

import pytest
from pydantic import ValidationError

from checkagent.datasets.schema import GoldenDataset, TestCase


class TestTestCase:
    """Tests for the TestCase model."""

    def test_minimal_case(self):
        case = TestCase(id="test-001", input="hello")
        assert case.id == "test-001"
        assert case.input == "hello"
        assert case.expected_tools == []
        assert case.expected_output_contains == []
        assert case.expected_output_equals is None
        assert case.max_steps is None
        assert case.tags == []

    def test_full_case(self):
        case = TestCase(
            id="refund-001",
            input="I want to return order #12345",
            expected_tools=["lookup_order", "check_return_policy", "initiate_refund"],
            expected_output_contains=["refund initiated", "3-5 business days"],
            expected_output_equals=None,
            max_steps=8,
            tags=["refund", "happy-path"],
            context={"user_id": "u123"},
            metadata={"priority": "high"},
        )
        assert case.expected_tools == ["lookup_order", "check_return_policy", "initiate_refund"]
        assert case.max_steps == 8
        assert len(case.tags) == 2
        assert case.context["user_id"] == "u123"

    def test_max_steps_must_be_positive(self):
        with pytest.raises(ValidationError, match="max_steps must be >= 1"):
            TestCase(id="bad", input="x", max_steps=0)

    def test_max_steps_negative(self):
        with pytest.raises(ValidationError, match="max_steps must be >= 1"):
            TestCase(id="bad", input="x", max_steps=-1)

    def test_id_required(self):
        with pytest.raises(ValidationError):
            TestCase(input="hello")  # type: ignore[call-arg]

    def test_input_required(self):
        with pytest.raises(ValidationError):
            TestCase(id="test-001")  # type: ignore[call-arg]

    def test_expected_output_equals(self):
        case = TestCase(id="t1", input="hi", expected_output_equals="hello there")
        assert case.expected_output_equals == "hello there"


class TestGoldenDataset:
    """Tests for the GoldenDataset model."""

    def test_minimal_dataset(self):
        ds = GoldenDataset(cases=[TestCase(id="t1", input="hi")])
        assert ds.name == "unnamed"
        assert ds.version == "1"
        assert len(ds.cases) == 1

    def test_full_dataset(self):
        ds = GoldenDataset(
            name="refund-tests",
            version="2",
            description="Tests for refund flow",
            cases=[
                TestCase(id="r1", input="refund me", tags=["refund"]),
                TestCase(id="r2", input="cancel order", tags=["cancel"]),
            ],
            metadata={"author": "test"},
        )
        assert ds.name == "refund-tests"
        assert len(ds.cases) == 2

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate test case IDs"):
            GoldenDataset(
                cases=[
                    TestCase(id="dup", input="a"),
                    TestCase(id="dup", input="b"),
                ]
            )

    def test_empty_cases_allowed(self):
        ds = GoldenDataset(cases=[])
        assert len(ds.cases) == 0

    def test_filter_by_tags(self):
        ds = GoldenDataset(
            cases=[
                TestCase(id="t1", input="a", tags=["refund", "happy"]),
                TestCase(id="t2", input="b", tags=["cancel"]),
                TestCase(id="t3", input="c", tags=["refund", "edge"]),
                TestCase(id="t4", input="d", tags=[]),
            ]
        )
        refund_cases = ds.filter_by_tags("refund")
        assert len(refund_cases) == 2
        assert {c.id for c in refund_cases} == {"t1", "t3"}

    def test_filter_by_multiple_tags(self):
        ds = GoldenDataset(
            cases=[
                TestCase(id="t1", input="a", tags=["refund"]),
                TestCase(id="t2", input="b", tags=["cancel"]),
                TestCase(id="t3", input="c", tags=[]),
            ]
        )
        result = ds.filter_by_tags("refund", "cancel")
        assert len(result) == 2

    def test_filter_no_matches(self):
        ds = GoldenDataset(cases=[TestCase(id="t1", input="a", tags=["x"])])
        assert ds.filter_by_tags("y") == []

    def test_get_case_found(self):
        ds = GoldenDataset(
            cases=[
                TestCase(id="t1", input="a"),
                TestCase(id="t2", input="b"),
            ]
        )
        case = ds.get_case("t2")
        assert case is not None
        assert case.input == "b"

    def test_get_case_not_found(self):
        ds = GoldenDataset(cases=[TestCase(id="t1", input="a")])
        assert ds.get_case("missing") is None

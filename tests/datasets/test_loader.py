"""Tests for golden dataset loader."""

import json

import pytest
from pydantic import ValidationError

from checkagent.datasets.loader import load_cases, load_dataset, parametrize_cases
from checkagent.datasets.schema import GoldenDataset, TestCase


@pytest.fixture
def bare_cases():
    """Bare list of test case dicts (no wrapper)."""
    return [
        {"id": "t1", "input": "hello", "tags": ["greet"]},
        {"id": "t2", "input": "bye", "tags": ["farewell"]},
        {"id": "t3", "input": "help", "expected_tools": ["search"], "tags": ["greet"]},
    ]


@pytest.fixture
def full_dataset(bare_cases):
    """Full dataset dict with metadata."""
    return {
        "name": "test-suite",
        "version": "2",
        "description": "Test golden dataset",
        "cases": bare_cases,
    }


@pytest.fixture
def json_bare_file(tmp_path, bare_cases):
    """JSON file with bare list format."""
    path = tmp_path / "bare.json"
    path.write_text(json.dumps(bare_cases))
    return path


@pytest.fixture
def json_full_file(tmp_path, full_dataset):
    """JSON file with full dataset format."""
    path = tmp_path / "full.json"
    path.write_text(json.dumps(full_dataset))
    return path


@pytest.fixture
def yaml_file(tmp_path, bare_cases):
    """YAML file with bare list format."""
    pytest.importorskip("yaml")
    import yaml

    path = tmp_path / "cases.yaml"
    path.write_text(yaml.dump(bare_cases))
    return path


class TestLoadDataset:
    """Tests for load_dataset()."""

    def test_load_json_bare(self, json_bare_file):
        ds = load_dataset(json_bare_file)
        assert isinstance(ds, GoldenDataset)
        assert len(ds.cases) == 3
        assert ds.name == "unnamed"

    def test_load_json_full(self, json_full_file):
        ds = load_dataset(json_full_file)
        assert ds.name == "test-suite"
        assert ds.version == "2"
        assert len(ds.cases) == 3

    def test_load_yaml(self, yaml_file):
        ds = load_dataset(yaml_file)
        assert len(ds.cases) == 3

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset("/nonexistent/path.json")

    def test_unsupported_format(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("id,input\nt1,hello")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset(path)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid json")
        with pytest.raises(json.JSONDecodeError):
            load_dataset(path)

    def test_dict_without_cases_key(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"name": "test", "items": []}))
        with pytest.raises(ValueError, match="must contain a 'cases' key"):
            load_dataset(path)

    def test_validation_error_propagates(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"id": "t1"}]))  # missing 'input'
        with pytest.raises(ValidationError):
            load_dataset(path)

    def test_duplicate_ids_in_file(self, tmp_path):
        path = tmp_path / "dupes.json"
        path.write_text(json.dumps([
            {"id": "same", "input": "a"},
            {"id": "same", "input": "b"},
        ]))
        with pytest.raises(Exception, match="Duplicate"):
            load_dataset(path)

    def test_yaml_without_pyyaml(self, tmp_path, monkeypatch):
        """Test that loading YAML without PyYAML gives a clear error."""
        path = tmp_path / "data.yaml"
        path.write_text("- id: t1\n  input: hi\n")
        # Temporarily make yaml import fail

        import checkagent.datasets.loader as loader_mod

        original = loader_mod._load_raw

        def patched_load(p):
            if p.suffix in (".yaml", ".yml"):
                import builtins

                real_import = builtins.__import__

                def fake_import(name, *args, **kwargs):
                    if name == "yaml":
                        raise ImportError("no yaml")
                    return real_import(name, *args, **kwargs)

                monkeypatch.setattr(builtins, "__import__", fake_import)
                try:
                    return original(p)
                finally:
                    monkeypatch.setattr(builtins, "__import__", real_import)
            return original(p)

        monkeypatch.setattr(loader_mod, "_load_raw", patched_load)
        with pytest.raises(ImportError, match="PyYAML"):
            load_dataset(path)


class TestLoadCases:
    """Tests for load_cases()."""

    def test_load_all(self, json_bare_file):
        cases = load_cases(json_bare_file)
        assert len(cases) == 3
        assert all(isinstance(c, TestCase) for c in cases)

    def test_filter_by_tag(self, json_bare_file):
        cases = load_cases(json_bare_file, tags=["greet"])
        assert len(cases) == 2
        assert {c.id for c in cases} == {"t1", "t3"}

    def test_filter_no_match(self, json_bare_file):
        cases = load_cases(json_bare_file, tags=["nonexistent"])
        assert cases == []


class TestParametrizeCases:
    """Tests for parametrize_cases()."""

    def test_returns_parametrize_tuple(self, json_bare_file):
        argname, argvalues = parametrize_cases(json_bare_file)
        assert argname == "test_case"
        assert len(argvalues) == 3

    def test_pytest_params_have_ids(self, json_bare_file):
        _, argvalues = parametrize_cases(json_bare_file)
        # pytest.param objects have an id attribute in .id
        for param in argvalues:
            assert hasattr(param, "id")

    def test_filter_with_tags(self, json_bare_file):
        _, argvalues = parametrize_cases(json_bare_file, tags=["farewell"])
        assert len(argvalues) == 1

    def test_cases_are_test_case_instances(self, json_bare_file):
        _, argvalues = parametrize_cases(json_bare_file)
        for param in argvalues:
            # pytest.param wraps values in .values tuple
            assert isinstance(param.values[0], TestCase)

"""Production trace import — convert observability traces into test cases.

Supports importing from JSON/JSONL files and OpenTelemetry OTLP JSON format.
Normalizes all traces to CheckAgent's AgentRun type, scrubs PII, and generates
test cases for regression testing.

Requirements: F6.2
"""

from checkagent.trace_import.base import TraceImporter
from checkagent.trace_import.json_importer import JsonFileImporter
from checkagent.trace_import.otel_importer import OtelJsonImporter
from checkagent.trace_import.pii import PiiScrubber
from checkagent.trace_import.testcase_gen import (
    TraceScreeningResult,
    generate_test_cases,
    screen_trace_output,
)

__all__ = [
    "TraceImporter",
    "JsonFileImporter",
    "OtelJsonImporter",
    "PiiScrubber",
    "TraceScreeningResult",
    "generate_test_cases",
    "screen_trace_output",
]

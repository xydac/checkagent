"""Benchmark cassette JSON serialization and parsing (RQ4 overhead data).

Measures parse/serialize times for cassettes of various sizes to fill
the metrics.md gaps for cassette performance.
"""

from __future__ import annotations

import time

from checkagent.replay.cassette import (
    Cassette,
    CassetteMeta,
    Interaction,
    RecordedRequest,
    RecordedResponse,
)


def _make_cassette(n_interactions: int) -> Cassette:
    """Create a cassette with n_interactions request-response pairs."""
    interactions = []
    for i in range(n_interactions):
        interactions.append(
            Interaction(
                sequence=i,
                request=RecordedRequest(
                    kind="llm",
                    method="chat.completions.create",
                    model="gpt-4o",
                    body={
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": f"Request {i}: explain quantum computing",
                            },
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1000,
                    },
                ),
                response=RecordedResponse(
                    status="ok",
                    body={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": (
                                        f"Response {i}: Quantum computing"
                                        " uses qubits that exist in"
                                        " multiple states simultaneously."
                                    ),
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 50 + i,
                            "completion_tokens": 80 + i,
                            "total_tokens": 130 + 2 * i,
                        },
                    },
                    prompt_tokens=50 + i,
                    completion_tokens=80 + i,
                    duration_ms=120.5 + i * 0.3,
                ),
            )
        )
    cassette = Cassette(
        meta=CassetteMeta(test_id=f"benchmark::test_{n_interactions}"),
        interactions=interactions,
    )
    cassette.finalize()
    return cassette


def test_cassette_json_serialize_100(benchmark_results: dict | None = None):
    """Benchmark serializing a 100-interaction cassette to JSON."""
    cassette = _make_cassette(100)

    # Warmup
    for _ in range(5):
        cassette.to_json()

    # Measure
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        cassette.to_json()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    json_str = cassette.to_json()
    size_kb = len(json_str.encode()) / 1024

    assert avg_ms < 150, f"Serialize took {avg_ms:.2f}ms, expected <150ms"
    # Expose results via print for collection
    print(f"\n[BENCHMARK] serialize_100: {avg_ms:.3f}ms avg, {size_kb:.1f}KB")


def test_cassette_json_parse_100():
    """Benchmark parsing a 100-interaction cassette from JSON."""
    cassette = _make_cassette(100)
    json_str = cassette.to_json()
    size_kb = len(json_str.encode()) / 1024

    # Warmup
    for _ in range(5):
        Cassette.from_json(json_str)

    # Measure
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        Cassette.from_json(json_str)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    assert avg_ms < 50, f"Parse took {avg_ms:.2f}ms, expected <50ms"
    print(f"\n[BENCHMARK] parse_100: {avg_ms:.3f}ms avg, {size_kb:.1f}KB")


def test_cassette_json_parse_10():
    """Benchmark parsing a 10-interaction cassette (typical test)."""
    cassette = _make_cassette(10)
    json_str = cassette.to_json()

    # Warmup
    for _ in range(5):
        Cassette.from_json(json_str)

    iterations = 200
    start = time.perf_counter()
    for _ in range(iterations):
        Cassette.from_json(json_str)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    assert avg_ms < 10, f"Parse took {avg_ms:.2f}ms, expected <10ms"
    print(f"\n[BENCHMARK] parse_10: {avg_ms:.3f}ms avg")


def test_cassette_json_parse_500():
    """Benchmark parsing a 500-interaction cassette (large session)."""
    cassette = _make_cassette(500)
    json_str = cassette.to_json()
    size_kb = len(json_str.encode()) / 1024

    iterations = 20
    start = time.perf_counter()
    for _ in range(iterations):
        Cassette.from_json(json_str)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    assert avg_ms < 200, f"Parse took {avg_ms:.2f}ms, expected <200ms"
    print(f"\n[BENCHMARK] parse_500: {avg_ms:.3f}ms avg, {size_kb:.1f}KB")


def test_cassette_finalize_overhead():
    """Measure finalize overhead (hashing, sequencing)."""
    cassette = _make_cassette(100)

    # Warmup
    for _ in range(5):
        cassette.finalize()

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        cassette.finalize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    assert avg_ms < 50, f"Finalize took {avg_ms:.2f}ms, expected <50ms"
    print(f"\n[BENCHMARK] finalize_100: {avg_ms:.3f}ms avg")


def test_cassette_integrity_check():
    """Measure verify_integrity overhead."""
    cassette = _make_cassette(100)
    cassette.finalize()

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        result = cassette.verify_integrity()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    assert result is True
    assert avg_ms < 50, f"Integrity check took {avg_ms:.2f}ms, expected <50ms"
    print(f"\n[BENCHMARK] integrity_100: {avg_ms:.3f}ms avg")

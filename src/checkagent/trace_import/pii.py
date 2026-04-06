"""PII scrubbing for imported production traces.

Replaces personally identifiable information with deterministic placeholders
so test logic is preserved while sensitive data is removed.

Uses configurable regex patterns. Optionally integrates with spaCy NER
for more thorough detection (opt-in, requires spaCy installation).

Requirements: F6.2
"""

from __future__ import annotations

import functools
import re
from typing import Any

# Default PII patterns with deterministic placeholder prefixes
_DEFAULT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")),
    ("IP_ADDR", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
]


class PiiScrubber:
    """Scrub PII from strings and nested data structures.

    Replacements are deterministic: the same input value always produces
    the same placeholder (e.g. first email found → <EMAIL_1>, second → <EMAIL_2>).
    """

    def __init__(
        self,
        *,
        extra_patterns: list[tuple[str, str]] | None = None,
        use_ner: bool = False,
    ) -> None:
        """Initialize the scrubber.

        Args:
            extra_patterns: Additional (label, regex_string) pairs to match.
            use_ner: If True, also use spaCy NER for PII detection.
                Requires spaCy and a language model to be installed.
        """
        self._patterns = list(_DEFAULT_PATTERNS)
        if extra_patterns:
            for label, pattern_str in extra_patterns:
                self._patterns.append((label, re.compile(pattern_str)))

        self._use_ner = use_ner
        self._counters: dict[str, int] = {}
        self._seen: dict[str, str] = {}

    def reset(self) -> None:
        """Reset placeholder counters for a new trace."""
        self._counters.clear()
        self._seen.clear()

    def scrub_text(self, text: str) -> str:
        """Replace PII in a string with deterministic placeholders."""
        if not text:
            return text

        result = text
        for label, pattern in self._patterns:
            repl = functools.partial(self._replace_match, label=label)
            result = pattern.sub(repl, result)

        if self._use_ner:
            result = self._ner_scrub(result)

        return result

    def scrub_value(self, value: Any) -> Any:
        """Recursively scrub PII from any JSON-compatible value."""
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, dict):
            return {k: self.scrub_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.scrub_value(v) for v in value]
        return value

    def _replace_match(self, m: re.Match[str], *, label: str) -> str:
        """Replacement callback for regex sub."""
        return self._placeholder(label, m.group())

    def _placeholder(self, label: str, original: str) -> str:
        """Get or create a deterministic placeholder for a PII value."""
        key = f"{label}:{original}"
        if key in self._seen:
            return self._seen[key]

        count = self._counters.get(label, 0) + 1
        self._counters[label] = count
        placeholder = f"<{label}_{count}>"
        self._seen[key] = placeholder
        return placeholder

    def _ner_scrub(self, text: str) -> str:
        """Use spaCy NER to detect and replace named entities."""
        try:
            import spacy  # type: ignore[import-untyped]
        except ImportError:
            return text

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            return text

        doc = nlp(text)
        # Process entities in reverse order to preserve offsets
        result = text
        for ent in reversed(doc.ents):
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC"):
                placeholder = self._placeholder(ent.label_, ent.text)
                result = result[: ent.start_char] + placeholder + result[ent.end_char :]

        return result

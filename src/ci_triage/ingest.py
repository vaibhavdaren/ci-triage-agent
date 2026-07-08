"""Ingest CI run logs and split them into per-test sections.

Works on tmt / pytest-style output. The parser is intentionally
heuristic: CI logs are messy, and a triage system must degrade
gracefully rather than fail on unexpected formats.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Patterns that mark the start of a test section in common runners.
TEST_START = re.compile(
    r"^(?:"
    r"::\s*(?P<tmt>/[\w/.-]+)"                      # tmt plan/test path
    r"|(?P<pytest>[\w/]+\.py::[\w\[\]-]+)\s+(?:PASSED|FAILED|ERROR|SKIPPED)"
    r"|=+\s*(?P<generic>[\w:. /-]+?)\s*=+$"
    r")"
)
RESULT_LINE = re.compile(r"\b(PASSED|FAILED|ERROR|SKIPPED|pass|fail|errr?or)\b")
# pytest banners like "=== FAILURES ===" are separators, not test sections;
# treating them as boundaries orphans tracebacks from their failing test.
PYTEST_BANNERS = {"failures", "errors", "warnings summary", "short test summary info",
                  "test session starts", "passed", "failed"}
FAIL_MARKERS = ("FAILED", "ERROR", "Traceback", "AssertionError", "fail")


@dataclass
class TestSection:
    run_id: str
    test_name: str
    text: str
    start_line: int
    end_line: int
    failed: bool = False
    error_excerpt: str = ""


@dataclass
class CIRun:
    run_id: str
    source: str
    sections: list[TestSection] = field(default_factory=list)

    @property
    def failed_sections(self) -> list[TestSection]:
        return [s for s in self.sections if s.failed]


def _extract_error_excerpt(text: str, max_lines: int = 12) -> str:
    """Pull the most diagnostic lines: traceback tail, assertion, stderr."""
    lines = text.splitlines()
    hits = [i for i, ln in enumerate(lines) if any(m in ln for m in FAIL_MARKERS)]
    if not hits:
        return "\n".join(lines[-max_lines:])
    anchor = hits[-1]
    lo = max(0, anchor - max_lines // 2)
    return "\n".join(lines[lo : anchor + max_lines // 2])


def parse_log(path: str | Path, run_id: str | None = None) -> CIRun:
    path = Path(path)
    run_id = run_id or path.stem
    raw = path.read_text(errors="replace")
    lines = raw.splitlines()

    run = CIRun(run_id=run_id, source=str(path))
    boundaries: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = TEST_START.match(line.strip())
        if m:
            generic = m.group("generic")
            if generic and generic.strip().lower() in PYTEST_BANNERS:
                continue  # separator banner — keep content with current section
            name = m.group("tmt") or m.group("pytest") or generic or f"section-{i}"
            # pytest lines embed the result; strip it from the name
            name = re.sub(r"\s+(PASSED|FAILED|ERROR|SKIPPED)\s*$", "", name)
            boundaries.append((i, name.strip()))

    if not boundaries:  # unstructured log -> single section
        boundaries = [(0, run_id)]

    for idx, (start, name) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start:end])
        failed = any(m in body for m in FAIL_MARKERS)
        run.sections.append(
            TestSection(
                run_id=run_id,
                test_name=name,
                text=body,
                start_line=start + 1,
                end_line=end,
                failed=failed,
                error_excerpt=_extract_error_excerpt(body) if failed else "",
            )
        )
    return run


def parse_directory(directory: str | Path) -> list[CIRun]:
    directory = Path(directory)
    runs = []
    for p in sorted(directory.glob("**/*")):
        if p.suffix in {".log", ".txt"} and p.is_file():
            runs.append(parse_log(p))
    return runs

"""Chunk test sections for retrieval.

Design decision worth defending in an interview: we chunk *within*
test sections rather than across the raw log, so every chunk carries
unambiguous test provenance (run_id + test_name). Log lines are short
and dense, so we chunk by lines with overlap instead of by tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ingest import CIRun


@dataclass
class Chunk:
    chunk_id: str
    run_id: str
    test_name: str
    text: str
    failed: bool


def chunk_runs(runs: list[CIRun], max_lines: int = 40, overlap: int = 8) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, max_lines - overlap)
    for run in runs:
        for sec in run.sections:
            lines = sec.text.splitlines()
            for start in range(0, max(1, len(lines)), step):
                window = lines[start : start + max_lines]
                if not window:
                    continue
                chunks.append(
                    Chunk(
                        # '|' separator: pytest names contain '::', so ':' is ambiguous
                        chunk_id=f"{run.run_id}|{sec.test_name}|{start}",
                        run_id=run.run_id,
                        test_name=sec.test_name,
                        text="\n".join(window),
                        failed=sec.failed,
                    )
                )
                if start + max_lines >= len(lines):
                    break
    return chunks

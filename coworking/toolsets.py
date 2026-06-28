from __future__ import annotations

import dataclasses
import pathlib
import re
from collections.abc import Iterable

import pydantic_ai

from coworking.structs import ChatDeps, ReadRecord


class Notebook:
    """A *private* stateful service: each agent gets its own instance."""

    def __init__(self) -> None:
        self._notes: list[str] = []

    def jot(self, note: str) -> str:
        """Append a note to your private notebook."""
        self._notes.append(note)
        return f"noted ({len(self._notes)} total)"

    def reread(self) -> str:
        """Reread all the notes you have jotted so far, numbered in order."""
        if not self._notes:
            return "no notes yet"
        return "\n".join(f"{i + 1}. {note}" for i, note in enumerate(self._notes))

    def toolset(self) -> pydantic_ai.FunctionToolset:
        ts = pydantic_ai.FunctionToolset()
        ts.add_function(self.jot)
        ts.add_function(self.reread)
        return ts


class Papers:
    """A shared read-only paper corpus service.

    All agents should share a single instance so they see the same files and
    indexing decisions stay consistent across the team.

    Tools take a ``paper_id`` (the file stem, e.g. ``2510.10185``) instead of
    a full filename, so agents never deal with paths or extensions. The
    ``read`` tool records what each agent has read into ``ctx.deps.reads`` so
    the team can see coverage and avoid duplicating work.
    """

    def __init__(self, root: pathlib.Path) -> None:
        self._root = root

    def list_files(self) -> str:
        """List the paper IDs (file stems) available in the corpus."""
        if not self._root.is_dir():
            return f"papers directory not found: {self._root}"
        files = sorted(p.stem for p in self._root.iterdir() if p.is_file() and p.suffix == ".md")
        if not files:
            return "no files"
        return "\n".join(files)

    def info(self, paper_id: str) -> str:
        """Get stats about a paper: character count, line count, word count, and section (heading) count."""
        path = self._path(paper_id)
        if path is None:
            return f"paper not found: {paper_id}"
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        section_count = sum(1 for line in lines if re.match(r"^#+\s", line))
        return (
            f"paper_id={paper_id}, chars={len(text)}, "
            f"lines={len(lines)}, words={len(text.split())}, sections={section_count}"
        )

    def read(
        self,
        ctx: pydantic_ai.RunContext[ChatDeps],
        paper_id: str,
        start_line: int = 0,
        end_line: int = 10,
    ) -> str:
        """Read lines [start_line, end_line) of a paper, with line numbers prefixed.

        Records the call into ``ctx.deps.reads`` so the team can see coverage
        and avoid duplicating reading effort.
        """
        path = self._path(paper_id)
        if path is None:
            return f"paper not found: {paper_id}"
        lines = path.read_text(encoding="utf-8").splitlines()
        start, end = self._clamp_range(start_line, end_line, len(lines))
        if start >= end:
            return f"empty range (start={start_line}, end={end_line}, total={len(lines)})"
        ctx.deps.reads.append(ReadRecord(paper_id=paper_id, start_line=start, end_line=end))
        return "\n".join(f"{i}: {line}" for i, line in enumerate(lines[start:end], start=start))

    def grep(
        self,
        paper_id: str,
        pattern: str,
        context: int = 2,
        regex: bool = False,
    ) -> str:
        """Search a paper for matches of a literal string or regex.

        Returns line numbers plus ``context`` lines before and after each match.
        Set ``regex=True`` to interpret ``pattern`` as a regular expression;
        otherwise it is matched literally.
        """
        path = self._path(paper_id)
        if path is None:
            return f"paper not found: {paper_id}"
        lines = path.read_text(encoding="utf-8").splitlines()
        matcher = self._compile(pattern, regex)
        if matcher is None:
            return f"invalid regex: {pattern!r}"
        return self._format_grep(lines, matcher, context)

    def toolset(self) -> pydantic_ai.FunctionToolset:
        ts = pydantic_ai.FunctionToolset()
        ts.add_function(self.list_files)
        ts.add_function(self.info)
        ts.add_function(self.read)
        ts.add_function(self.grep)
        return ts

    def _path(self, paper_id: str) -> pathlib.Path | None:
        if not paper_id or "/" in paper_id or "\\" in paper_id:
            return None
        path = self._root / f"{paper_id}.md"
        return path if path.is_file() else None

    @staticmethod
    def _clamp_range(start_line: int, end_line: int, total: int) -> tuple[int, int]:
        return max(0, start_line), min(total, end_line)

    @staticmethod
    def _compile(pattern: str, regex: bool) -> re.Pattern[str] | None:
        try:
            return re.compile(pattern if regex else re.escape(pattern))
        except re.error:
            return None

    @staticmethod
    def _format_grep(lines: list[str], matcher: re.Pattern[str], context: int) -> str:
        matches = [i for i, line in enumerate(lines) if matcher.search(line)]
        if not matches:
            return "no matches"
        blocks: list[str] = []
        for match_line in matches:
            start = max(0, match_line - context)
            end = min(len(lines), match_line + context + 1)
            rendered: Iterable[str] = (
                f"{j}:{'>' if j == match_line else ' '} {lines[j]}" for j in range(start, end)
            )
            blocks.append("\n".join(rendered))
        return "\n---\n".join(blocks)


@dataclasses.dataclass(frozen=True)
class ReportRecord:
    author: str
    content: str


class ReportBoard:
    """Shared service that collects the team's final reports.

    The team is expected to coordinate (via ``send_message``) so that only one
    agent publishes the consensus report. Agents who disagree with the
    consensus may publish dissenting reports separately. Every call is
    recorded in ``reports``.
    """

    def __init__(self) -> None:
        self._reports: list[ReportRecord] = []

    def publish(self, author: str, content: str) -> ReportRecord:
        record = ReportRecord(author=author, content=content)
        self._reports.append(record)
        return record

    @property
    def reports(self) -> list[ReportRecord]:
        return list(self._reports)

#!/usr/bin/env python3
"""Check the repository's human-documentation contract without dependencies."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"

ALLOWED_DOCS_ENTRIES = {
    "README.md",
    "architecture",
    "implementation",
    "developer",
    "user",
}
LEGACY_DOCS_DIRECTORIES = {
    "debug",
    "evidence",
    "guide",
    "milestones",
    "performance",
    "setup",
}
REQUIRED_READMES = (
    "AGENTS.md",
    "docs/README.md",
    "docs/architecture/README.md",
    "docs/implementation/README.md",
    "docs/implementation/tasks/README.md",
    "docs/developer/README.md",
    "docs/user/README.md",
    "evidence/README.md",
)

INLINE_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*([^\n)]*)\)")
REFERENCE_LINK_RE = re.compile(
    r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>\n]+>|\S+)", re.MULTILINE
)
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.*?)(?<!`)\1(?!`)")
URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def relative_display(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def without_fenced_code(text: str) -> str:
    """Blank fenced blocks while retaining line numbers for diagnostics."""
    output: list[str] = []
    fence_char = ""
    fence_length = 0

    for line in text.splitlines(keepends=True):
        match = FENCE_RE.match(line)
        if not fence_char:
            if match:
                marker = match.group(1)
                fence_char = marker[0]
                fence_length = len(marker)
                output.append("\n" if line.endswith(("\n", "\r")) else "")
            else:
                output.append(line)
            continue

        output.append("\n" if line.endswith(("\n", "\r")) else "")
        if match:
            marker = match.group(1)
            if marker[0] == fence_char and len(marker) >= fence_length:
                fence_char = ""
                fence_length = 0

    return "".join(output)


def without_inline_code(text: str) -> str:
    """Blank ordinary inline code spans while retaining source positions."""
    return INLINE_CODE_RE.sub(lambda match: " " * len(match.group(0)), text)


def link_destination(body: str) -> str:
    """Return a Markdown link destination, excluding an optional title."""
    body = body.strip()
    if not body:
        return ""
    if body.startswith("<"):
        closing = body.find(">", 1)
        return body[1:closing] if closing >= 0 else body[1:]

    destination: list[str] = []
    escaped = False
    for char in body:
        if escaped:
            destination.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char.isspace():
            break
        else:
            destination.append(char)
    if escaped:
        destination.append("\\")
    return "".join(destination)


def markdown_files() -> list[Path]:
    paths: set[Path] = set()
    for name in ("README.md", "AGENTS.md"):
        root_page = REPO_ROOT / name
        if root_page.is_file():
            paths.add(root_page)

    for directory in ("docs", "evidence", "scripts", "samples"):
        root = REPO_ROOT / directory
        if not root.is_dir():
            continue
        paths.update(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".md"
        )
    return sorted(paths, key=lambda path: relative_display(path).lower())


def extracted_links(text: str) -> list[tuple[int, str]]:
    visible = without_inline_code(without_fenced_code(text))
    links: list[tuple[int, str]] = []
    occupied: set[tuple[int, int]] = set()

    for match in INLINE_LINK_RE.finditer(visible):
        occupied.add(match.span())
        links.append(
            (
                visible.count("\n", 0, match.start()) + 1,
                link_destination(match.group(1)),
            )
        )

    for match in REFERENCE_LINK_RE.finditer(visible):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        links.append(
            (
                visible.count("\n", 0, match.start()) + 1,
                link_destination(match.group(1)),
            )
        )
    return links


def check_structure(errors: list[str]) -> None:
    if not DOCS_ROOT.is_dir():
        errors.append("docs/: required documentation directory is missing")
        return

    actual_entries = {entry.name for entry in DOCS_ROOT.iterdir()}
    for name in sorted(actual_entries - ALLOWED_DOCS_ENTRIES - LEGACY_DOCS_DIRECTORIES):
        errors.append(f"docs/{name}: unexpected top-level entry")
    for name in sorted(ALLOWED_DOCS_ENTRIES - actual_entries):
        errors.append(f"docs/{name}: required top-level entry is missing")

    for name in sorted(LEGACY_DOCS_DIRECTORIES):
        if (DOCS_ROOT / name).exists():
            errors.append(f"docs/{name}: retired documentation directory must not exist")

    for relative in REQUIRED_READMES:
        if not (REPO_ROOT / relative).is_file():
            errors.append(f"{relative}: required navigation README is missing")

    user_root = DOCS_ROOT / "user"
    if user_root.is_dir():
        retired_c_guides = sorted(
            path
            for path in user_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".md"
            and path.name.lower().startswith("c-")
        )
        for path in retired_c_guides:
            errors.append(
                f"{relative_display(path)}: retired C-language user guide is forbidden"
            )


def check_retired_ffi_symbol(errors: list[str]) -> None:
    candidates = [REPO_ROOT / "README.md"]
    user_root = DOCS_ROOT / "user"
    if user_root.is_dir():
        candidates.extend(sorted(user_root.rglob("*.md")))

    for path in candidates:
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "hyle_ffi_abi_version" in line:
                errors.append(
                    f"{relative_display(path)}:{line_number}: retired FFI symbol is "
                    "forbidden in user-facing documentation"
                )


def check_relative_links(paths: list[Path], errors: list[str]) -> int:
    checked = 0
    root_resolved = REPO_ROOT.resolve()

    for source in paths:
        text = source.read_text(encoding="utf-8")
        for line_number, destination in extracted_links(text):
            if not destination:
                continue
            if destination.startswith(("#", "//")) or URI_SCHEME_RE.match(destination):
                continue

            parsed = urlsplit(destination)
            relative_path = unquote(parsed.path)
            if not relative_path:
                continue
            checked += 1

            if relative_path.startswith("/"):
                candidate = REPO_ROOT / relative_path.lstrip("/")
            else:
                candidate = source.parent / relative_path
            candidate = candidate.resolve(strict=False)

            try:
                candidate.relative_to(root_resolved)
            except ValueError:
                errors.append(
                    f"{relative_display(source)}:{line_number}: relative link escapes "
                    f"the repository: {destination!r}"
                )
                continue

            if not candidate.exists():
                errors.append(
                    f"{relative_display(source)}:{line_number}: broken relative link "
                    f"{destination!r} -> {relative_display(candidate)}"
                )
    return checked


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    errors: list[str] = []
    check_structure(errors)
    check_retired_ffi_symbol(errors)
    files = markdown_files()
    links = check_relative_links(files, errors)

    if errors:
        print(f"documentation contract failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        f"documentation contract passed: {len(files)} Markdown files, "
        f"{links} relative links checked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

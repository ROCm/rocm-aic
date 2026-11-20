#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Generate the variable tables in docs/ENVIRONMENT.md from the files that own the defaults.

Each default is read from the file that defines it:

  docker/docker-bake.hcl      `variable` blocks (build inputs)
  docker/docker-compose.yml   `${VAR:-default}` / `${VAR-default}` interpolations
  Makefile, mk/*.mk           top-level `VAR ?= value` lines, plus the SPUR block
  Slurm and shell scripts     `${VAR:-default}`, `${VAR:=default}`, `${VAR:?}`
  Python tests/benchmarks     `os.getenv("VAR", "default")`

Bake and compose own whatever they declare; otherwise the first file in that
order with a non-empty default is the variable's owner (OWNER_OVERRIDES names
the few exceptions).  A different default
at another entry point is listed after it.

Descriptions and the grouping into tables are hand-written in
docs/environment-vars.txt and survive regeneration.  --check fails when the
tables are stale, when a variable found in code has no entry there, or when an
entry names a variable that is no longer in code.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC = Path("docs/ENVIRONMENT.md")
DESCRIPTIONS = Path("docs/environment-vars.txt")
BAKE_FILE = Path("docker/docker-bake.hcl")
COMPOSE_FILE = Path("docker/docker-compose.yml")
MAKE_FILES = ("Makefile", "mk/local.mk", "mk/bench.mk", "mk/dist.mk", "mk/infra.mk")
# Order matters: the first file with a non-empty default owns the variable.  The
# two sourced libraries come before the scripts that source them.
SHELL_FILES = (
    "monitoring/monitoring-lib.sh",
    "scripts/aic-endpoints.sh",
    ".slurm/run-build-distribute.sh",
    ".slurm/run-cliff.sbatch",
    ".slurm/run-accuracy.sh",
    ".slurm/run-kvbench-cliff.sbatch",
    ".slurm/ext-hit-test.sbatch",
    ".slurm/test-nightly-wheels.sbatch",
    ".slurm/image-dist-lib.sh",
    ".github/scripts/run-vllm-emul-perf.sh",
    ".github/scripts/runners/spur-ci-common.sh",
    "benchmarks/agentx/run_agentx.sh",
    "benchmarks/agentx/scaled_corpus.sh",
    "benchmarks/agentx/overnight_sweep.sh",
)
PYTHON_FILES = (
    "benchmarks/common/kv_client.py",
    "benchmarks/aic_bench_serve.py",
    "benchmarks/aic_bench.py",
    "benchmarks/vllm_reset_test.py",
    "tests/accuracy/test_accuracy.py",
    "tests/accuracy/conftest.py",
)

# Where scan order would name a file that only carries a fallback.
OWNER_OVERRIDES = {
    "AIC_SPUR_PARTITION": "Makefile",
    "AIC_SHARED_NFS": "Makefile",
    "AIC_IMAGE": ".slurm/run-build-distribute.sh",
}

IGNORE_SECTION = "ignore"
# Set by the scheduler, the CI runner or the shell, never by a user of this repo.
FOREIGN_PREFIXES = ("SLURM_", "SPUR_", "GITHUB_", "RUNNER_", "BASH")
SPUR_CONDITION = "ifeq ($(AIC_SPUR_CLUSTER),1)"
# A script that reads the owner's default through a helper holds no copy.
OWNER_HELPERS = ("aic-compose-default.sh", "aic-bake-env.sh")
# A `${VAR:-text}` inside a message is what the message prints, not a default.
MESSAGE_LINE = re.compile(r"^\s*(?:log|echo|printf|warn|die)\b")
NAME = r"[A-Z][A-Z0-9_]*"
MARKER = re.compile(
    r"^<!-- BEGIN generated:(?P<id>[a-z0-9-]+) -->\n(?P<body>.*?)^<!-- END generated:(?P=id) -->$",
    re.M | re.S,
)


class GenerateError(Exception):
    """Raised when the reference cannot be rendered from the sources."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


@dataclass(frozen=True)
class Site:
    """One place a variable gets a default."""

    name: str
    default: str | None  # None: no default at this site
    path: str
    note: str = ""
    # `${VAR-x}` / `${VAR=x}`: the default applies only when VAR is unset, so
    # an empty value is a setting of its own.
    keeps_empty: bool = False


def report_error(path: Path, message: str) -> None:
    print(f"ERROR: {path}: {message}", file=sys.stderr)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        annotation = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error file={path}::{annotation}", file=sys.stderr)


def read(root: Path, path: Path | str) -> str:
    return (root / path).read_text(encoding="utf-8")


def code_lines(text: str) -> str:
    """The text with every whole-line `#` comment blanked (line numbers kept)."""
    return "\n".join("" if line.lstrip().startswith("#") else line for line in text.split("\n"))


def expansions(text: str, escape: str = ""):
    """Yield (offset, name, operator, argument) for each `${NAME...}` in text.

    `escape` is the character that, placed before the `$`, makes the expansion
    someone else's: compose passes `$${VAR}` through to the container shell.
    Nested expansions are yielded too, after the one that contains them.
    """
    start = 0
    while True:
        start = text.find("${", start)
        if start < 0:
            return
        if escape and start > 0 and text[start - 1] == escape:
            start += 2
            continue
        depth = 0
        end = start + 1
        while end < len(text):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        match = re.match(rf"({NAME})(:?[-=?+])?(.*)\Z", text[start + 2 : end], re.S)
        if match:
            yield start, match.group(1), match.group(2) or "", match.group(3)
        start += 2


# ---- docker/docker-bake.hcl -------------------------------------------------


def bake_sites(root: Path) -> list[Site]:
    text = read(root, BAKE_FILE)
    sites: list[Site] = []
    literals: dict[str, str] = {}
    for match in re.finditer(r'^variable "([A-Za-z_][A-Za-z0-9_]*)"\s*\{(.*)$', text, re.M):
        name, rest = match.group(1), match.group(2)
        if name.startswith("_"):
            continue
        if rest.strip():
            # One line: `variable "X" { default = <expression> }  # comment`
            default = re.fullmatch(r"\s*default\s*=\s*(.*)\}\s*(?:#.*)?", rest)
        else:
            body = text[match.end() : text.index("\n}", match.end())]
            default = re.search(r"^\s*default\s*=\s*(.*)\Z", body, re.M | re.S)
        if not default:
            raise GenerateError(BAKE_FILE, f"cannot read the default of variable {name}")
        expression = " ".join(default.group(1).split())
        literal = re.fullmatch(r'"([^"$\\]*)"', expression)
        if literal:
            literals[name] = literal.group(1)
            sites.append(Site(name, literal.group(1), str(BAKE_FILE)))
        else:
            template = re.fullmatch(r'"([^"\\]*)"', expression)
            sites.append(Site(name, template.group(1) if template else expression, str(BAKE_FILE), "computed"))
    verify_bake(root, {site.name for site in sites}, literals)
    return sites


def verify_bake(root: Path, names: set[str], literals: dict[str, str]) -> None:
    """Hold the HCL parse to what bake itself reports, where bake can say.

    `--list=variables`, not `--print`: it reports the variables themselves
    rather than the targets they resolve into.
    """
    env = {key: value for key, value in os.environ.items() if key not in names}
    try:
        result = subprocess.run(
            ["docker", "buildx", "bake", "-f", str(root / BAKE_FILE), "--list=type=variables,format=json"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
    except OSError:
        return
    if result.returncode != 0:
        # No buildx, or a bake that predates --list: the parse stands alone.
        return
    try:
        listed = {entry["name"]: entry.get("value", "") for entry in json.loads(result.stdout)}
    except (ValueError, KeyError, TypeError):
        return
    listed = {name: value for name, value in listed.items() if not name.startswith("_")}
    if set(listed) != names:
        raise GenerateError(
            BAKE_FILE,
            "variable blocks parsed differently from `docker buildx bake --list=variables`: "
            + ", ".join(sorted(set(listed) ^ names)),
        )
    for name, value in literals.items():
        if listed[name] != value:
            raise GenerateError(BAKE_FILE, f"{name}: parsed default {value!r}, bake reports {listed[name]!r}")


# ---- docker/docker-compose.yml ----------------------------------------------


def compose_sites(root: Path) -> list[Site]:
    text = code_lines(read(root, COMPOSE_FILE))
    blocks: list[tuple[int, str]] = []
    in_services = False
    for match in re.finditer(r"^(?:(\S[^:\n]*):|  ([A-Za-z0-9_.-]+):\s*$)", text, re.M):
        if match.group(1) is not None:
            in_services = match.group(1) == "services"
            if not in_services:
                blocks.append((match.start(), match.group(1)))
        elif in_services:
            blocks.append((match.start(), match.group(2)))
    sites: list[Site] = []
    plain: dict[str, Site] = {}
    for offset, name, operator, argument in expansions(text, escape="$"):
        block = next((label for start, label in reversed(blocks) if start <= offset), "")
        if operator in (":-", "-"):
            sites.append(Site(name, argument, str(COMPOSE_FILE), block, operator == "-"))
        elif operator in ("", "?", ":?"):
            plain.setdefault(name, Site(name, None, str(COMPOSE_FILE), block))
    # A variable compose only reads still needs an entry in the reference.
    defaulted = {site.name for site in sites}
    return sites + [site for name, site in plain.items() if name not in defaulted]


# ---- Makefile, mk/*.mk ------------------------------------------------------


def make_sites(root: Path, path: str) -> list[Site]:
    lines = read(root, path).split("\n")
    sites: list[Site] = []
    conditions: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.startswith("\t") or line.lstrip().startswith("#"):
            continue
        while line.endswith("\\") and index < len(lines):
            line = line[:-1].rstrip() + " " + lines[index].strip()
            index += 1
        stripped = line.strip()
        if re.match(r"(ifeq|ifneq|ifdef|ifndef)\b", stripped):
            conditions.append(stripped)
            continue
        if re.match(r"else\b", stripped):
            if conditions:
                conditions[-1] = "else"
            continue
        if stripped == "endif":
            if conditions:
                conditions.pop()
            continue
        match = re.match(rf"(?:override\s+)?(?:export\s+)?({NAME})\s*(\?=|:=)\s*(.*)$", stripped)
        if not match:
            continue
        name, operator, value = match.groups()
        value = re.sub(r"\s+#.*$", "", value).strip()
        spur = SPUR_CONDITION in conditions
        if operator == ":=" and not spur:
            continue
        note = "SPUR" if spur else ("conditional" if conditions else "")
        sites.append(Site(name, value, path, note))
    return sites


# ---- Slurm and shell scripts ------------------------------------------------


def shell_sites(root: Path, path: str) -> list[Site]:
    text = code_lines(read(root, path))
    sites: list[Site] = []
    seen: set[tuple[str, str | None]] = set()
    for offset, name, operator, argument in expansions(text):
        line_start = text.rfind("\n", 0, offset) + 1
        line_end = text.find("\n", offset)
        line = text[line_start : line_end if line_end >= 0 else len(text)]
        if operator in (":-", "-", ":=", "="):
            if MESSAGE_LINE.match(line) or re.fullmatch(r"<[^<>]*>", argument):
                continue
            default: str | None = argument
            note = ""
        elif operator in (":?", "?"):
            default, note = None, "required"
        else:
            continue
        if (name, default) in seen:
            continue
        first = not any(site.name == name for site in sites)
        seen.add((name, default))
        note = note if first or note else "another branch"
        sites.append(Site(name, default, path, note, operator in ("-", "=")))
    return sites


# ---- Python tests and benchmarks --------------------------------------------


def python_sites(root: Path, path: str) -> list[Site]:
    text = read(root, path)
    sites: list[Site] = []
    pattern = re.compile(
        rf"""os\.(?:getenv|environ\.get)\(\s*["']({NAME})["']\s*(?:,\s*(["'])(.*?)\2\s*,?\s*)?\)"""
        rf"""|os\.environ\[["']({NAME})["']\]""",
        re.S,
    )
    for match in pattern.finditer(text):
        if match.group(4):
            sites.append(Site(match.group(4), None, path, "required"))
        else:
            sites.append(Site(match.group(1), match.group(3), path))
    return sites


# ---- Ownership ----------------------------------------------------------------


def collect(root: Path) -> dict[str, list[Site]]:
    """Every default site of every variable, in ownership order."""
    sites = bake_sites(root) + compose_sites(root)
    for path in MAKE_FILES:
        sites += make_sites(root, path)
    for path in SHELL_FILES:
        sites += shell_sites(root, path)
    for path in PYTHON_FILES:
        sites += python_sites(root, path)
    variables: dict[str, list[Site]] = {}
    for site in sites:
        if site.name.startswith(FOREIGN_PREFIXES):
            continue
        variables.setdefault(site.name, []).append(site)
    return variables


def owner(sites: list[Site]) -> Site:
    override = OWNER_OVERRIDES.get(sites[0].name)
    for site in sites:
        if site.path == override and site.default is not None:
            return site
    # Bake and compose own whatever they declare, an empty default included.
    if sites[0].path == str(BAKE_FILE) or (sites[0].path == str(COMPOSE_FILE) and sites[0].default is not None):
        return sites[0]
    for site in sites:
        if site.default and site.note not in ("SPUR", "conditional", "another branch"):
            return site
    for site in sites:
        if site.default is not None:
            return site
    return sites[0]


def reads_owner(default: str) -> bool:
    """True for a default that defers to another variable or to the owner's helper."""
    if any(helper in default for helper in OWNER_HELPERS):
        return True
    return re.fullmatch(rf"\$[{{(]{NAME}(:-)?[}})]", default) is not None


def mentioned_in(root: Path, name: str) -> str | None:
    """The first scanned file whose code mentions NAME, for variables with no default anywhere."""
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])")
    for path in (*SHELL_FILES, *PYTHON_FILES, *MAKE_FILES, str(COMPOSE_FILE), str(BAKE_FILE)):
        if pattern.search(code_lines(read(root, path))):
            return path
    return None


# ---- Descriptions -------------------------------------------------------------


def load_descriptions(root: Path) -> dict[str, dict[str, str]]:
    """Section id -> {variable: description}, both in file order."""
    sections: dict[str, dict[str, str]] = {}
    section: dict[str, str] | None = None
    last: str | None = None
    for number, line in enumerate(read(root, DESCRIPTIONS).split("\n"), 1):
        where = f"line {number}"
        if not line.strip() or line.startswith("#"):
            continue
        header = re.fullmatch(r"\[([a-z0-9-]+)\]", line.strip())
        if header:
            if header.group(1) in sections:
                raise GenerateError(DESCRIPTIONS, f"{where}: section [{header.group(1)}] appears twice")
            section = sections.setdefault(header.group(1), {})
            last = None
            continue
        if section is None:
            raise GenerateError(DESCRIPTIONS, f"{where}: entry before the first [section]")
        if line[0].isspace():
            if last is None:
                raise GenerateError(DESCRIPTIONS, f"{where}: continuation line without an entry")
            section[last] = f"{section[last]} {line.strip()}".strip()
            continue
        entry = re.fullmatch(rf"({NAME}):(?:\s+(.*))?", line.rstrip())
        if not entry:
            raise GenerateError(DESCRIPTIONS, f"{where}: expected `NAME: description`, found {line!r}")
        last = entry.group(1)
        if any(last in entries for entries in sections.values()):
            raise GenerateError(DESCRIPTIONS, f"{where}: {last} is listed twice")
        section[last] = (entry.group(2) or "").strip()
    return sections


# ---- Rendering ----------------------------------------------------------------


def code(value: str) -> str:
    value = " ".join(value.split()).replace("|", "\\|")
    if "`" in value:
        return f"`` {value} ``"
    return f"`{value}`"


def default_cell(sites: list[Site]) -> str:
    primary = owner(sites)
    if primary.default is None:
        cell = "(required)" if primary.note == "required" else "(unset)"
    elif primary.default == "":
        cell = "(empty)"
    elif primary.note == "computed" and len(primary.default) > 80:
        cell = "(computed)"
    else:
        cell = code(primary.default) + (" (only when unset)" if primary.keeps_empty else "")
    if primary.note in ("SPUR", "conditional"):
        cell += f" ({primary.note})"
    shown = {primary.default}
    for site in sites:
        if site is primary or not site.default or reads_owner(site.default):
            continue
        # A site default is always worth a mention; other repeats are not.
        if site.default == primary.default or (site.default in shown and site.note != "SPUR"):
            continue
        shown.add(site.default)
        if site.path == primary.path:
            label = code(site.note) if site.path == str(COMPOSE_FILE) and site.note else site.note or "another branch"
        else:
            label = code(site.path) + (f", {site.note}" if site.note in ("SPUR", "conditional") else "")
        cell += f"; {code(site.default)} ({label})"
    return cell


def render_tables(root: Path) -> dict[str, str]:
    variables = collect(root)
    sections = load_descriptions(root)
    listed = {name for entries in sections.values() for name in entries}

    missing = sorted(set(variables) - listed)
    if missing:
        detail = ", ".join(f"{name} ({owner(variables[name]).path})" for name in missing)
        raise GenerateError(
            DESCRIPTIONS,
            f"no entry for: {detail}. Add `NAME: description` under the section whose table it "
            f"belongs in, or under [{IGNORE_SECTION}] if it is not a user-facing variable.",
        )

    tables: dict[str, str] = {}
    for section, entries in sections.items():
        if section == IGNORE_SECTION:
            stale = sorted(name for name in entries if name not in variables and not mentioned_in(root, name))
            if stale:
                raise GenerateError(DESCRIPTIONS, f"[{IGNORE_SECTION}] names variables no longer in code: {', '.join(stale)}")
            continue
        rows = ["| Variable | Default | Owner | Description |", "| --- | --- | --- | --- |"]
        for name, description in entries.items():
            if re.search(r"(?<!\\)\|", description):
                raise GenerateError(DESCRIPTIONS, f"{name}: write '|' as '\\|' in a description")
            if name in variables:
                sites = variables[name]
                rows.append(f"| `{name}` | {default_cell(sites)} | `{owner(sites).path}` | {description} |")
                continue
            path = mentioned_in(root, name)
            if path is None:
                raise GenerateError(DESCRIPTIONS, f"{name} is described but no longer appears in code; remove the entry")
            rows.append(f"| `{name}` | (unset) | `{path}` | {description} |")
        tables[section] = "\n".join(rows) + "\n"
    return tables


def render_doc(root: Path, doc_text: str) -> str:
    tables = render_tables(root)
    found = [match.group("id") for match in MARKER.finditer(doc_text)]
    for section in tables:
        if found.count(section) != 1:
            raise GenerateError(
                DOC,
                f"expected exactly one `<!-- BEGIN generated:{section} -->` ... "
                f"`<!-- END generated:{section} -->` pair, found {found.count(section)}",
            )
    unknown = sorted(set(found) - set(tables))
    if unknown:
        raise GenerateError(DOC, f"generated markers with no section in {DESCRIPTIONS}: {', '.join(unknown)}")
    return MARKER.sub(
        lambda match: f"<!-- BEGIN generated:{match.group('id')} -->\n{tables[match.group('id')]}"
        f"<!-- END generated:{match.group('id')} -->",
        doc_text,
    )


def dump(root: Path) -> None:
    for name, sites in sorted(collect(root).items()):
        print(f"{name}\t{owner(sites).path}\t{default_cell(sites)}")


def check_or_write(root: Path, write: bool) -> int:
    doc_text = read(root, DOC)
    rendered = render_doc(root, doc_text)
    if rendered == doc_text:
        print(f"{DOC} variable tables are in sync.")
        return 0
    if write:
        (root / DOC).write_text(rendered, encoding="utf-8")
        print(f"Updated {DOC}")
        return 0
    report_error(DOC, "generated variable tables are out of sync with the files that own the defaults")
    sys.stdout.writelines(
        difflib.unified_diff(
            doc_text.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=f"{DOC} (committed)",
            tofile=f"{DOC} (expected)",
        )
    )
    print(f"Run `make env-docs`, then commit {DOC}.", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help=f"fail if {DOC} is stale")
    mode.add_argument("--write", action="store_true", help=f"update {DOC} in place")
    mode.add_argument("--dump", action="store_true", help="print every variable found: name, owner, default")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        if args.dump:
            dump(args.root)
            return 0
        return check_or_write(args.root, write=args.write)
    except (OSError, UnicodeError) as error:
        path = Path(error.filename) if getattr(error, "filename", None) else args.root
        report_error(path, str(error))
    except GenerateError as error:
        report_error(error.path, str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

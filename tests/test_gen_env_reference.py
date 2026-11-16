# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""docs/ENVIRONMENT.md is generated from the files that own the defaults.

Covers scripts/gen-env-reference.py: each extractor, the ownership rule, the
description file, and the committed reference itself.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gen-env-reference.py"

_spec = importlib.util.spec_from_file_location("gen_env_reference", SCRIPT)
assert _spec and _spec.loader
gen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gen
_spec.loader.exec_module(gen)

BAKE = """\
variable "PIN" { default = "v1.2.3" }
variable "KNOB"   { default = "" }       # empty = nproc
variable "BASE" { default = "repo/image:${PIN}-full" }
variable "SHORT" { default = substr(PIN, 0, 2) }
variable "_INTERNAL" {
  default = "${PIN}-x"
}
variable "TAG" {
  default = PIN == "" ? _INTERNAL : "${_INTERNAL}-y"
}
"""

COMPOSE = """\
# ${COMMENTED:-no}
x-env: &env
  PORT: "${PORT:-6555}"
services:
  server:
    image: ${IMAGE_REF:-server:latest}
    pid: "${PID_MODE-service:other}"
    command: |
      run --port "$${PORT:-1}" --model ${MODEL:-big} --gpu ${ROCR:-${GPU:-0}}
  emulator:
    command: run --model ${MODEL:-small} ${MODEL:-small} ${MODEL}
    volumes:
      - ${LOGDIR}:/logs
"""

MAKEFILE = """\
PORT ?= 6555
MODEL ?=
LOGDIR ?= $(CURDIR)/logs  # where logs go
_PRIVATE ?= 1
FIXED := not-a-default
LONG ?= a \\
        b
ifeq ($(AIC_SPUR_CLUSTER),1)
export PARTITION ?= spur-v2
export LOGDIR := /shared/logs
else
OTHER ?= elsewhere
endif
target:
\tRECIPE ?= no
"""

SHELL = """\
#!/bin/bash
# COMMENTED="${COMMENTED:-no}"
TIME="${TIME:-04:00:00}"
: "${MONITORING:=1}"
: "${PID_MODE=host}"
NEEDED="${NEEDED:?must be set}"
MODEL="${MODEL:-}"
if spur; then DRIVER="${DRIVER:-docker}"; else DRIVER="${DRIVER:-docker-container}"; fi
log "limit ${LIMIT:-full split} job ${SLURM_JOB_ID:-manual}"
echo "${SHOWN:-<none>}"
IMAGE="${IMAGE_REF:-$(bash docker/scripts/aic-compose-default.sh IMAGE_REF)}"
PARTITION="${PARTITION:-defq}"
"""

PYTHON = """\
import os
URL = os.getenv("TEST_URL",   "http://server:8000")
N = int(os.environ.get('TEST_N', '10'))
LIMIT = os.getenv("TEST_LIMIT")
DIR = os.environ["TEST_DIR"]
"""


def _root(tmp: str, **files: str) -> Path:
    """A tree holding every file the generator reads, empty unless given."""
    root = Path(tmp)
    paths = [gen.BAKE_FILE, gen.COMPOSE_FILE, *gen.MAKE_FILES, *gen.SHELL_FILES, *gen.PYTHON_FILES]
    for path in map(str, paths):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(files.get(path, ""), encoding="utf-8")
    (root / "docs").mkdir(exist_ok=True)
    return root


def _defaults(sites: list) -> dict[str, list[str | None]]:
    found: dict[str, list[str | None]] = {}
    for site in sites:
        found.setdefault(site.name, []).append(site.default)
    return found


class ExtractorTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = _root(
            tmp.name,
            **{
                str(gen.BAKE_FILE): BAKE,
                str(gen.COMPOSE_FILE): COMPOSE,
                "Makefile": MAKEFILE,
                gen.SHELL_FILES[0]: SHELL,
                gen.PYTHON_FILES[0]: PYTHON,
            },
        )

    def test_bake_variable_blocks(self) -> None:
        sites = {site.name: site for site in gen.bake_sites(self.root)}
        self.assertEqual(sorted(sites), ["BASE", "KNOB", "PIN", "SHORT", "TAG"])
        self.assertEqual((sites["PIN"].default, sites["PIN"].note), ("v1.2.3", ""))
        self.assertEqual(sites["KNOB"].default, "")
        self.assertEqual((sites["BASE"].default, sites["BASE"].note), ("repo/image:${PIN}-full", "computed"))
        self.assertEqual(sites["SHORT"].default, "substr(PIN, 0, 2)")
        self.assertEqual(sites["TAG"].default, 'PIN == "" ? _INTERNAL : "${_INTERNAL}-y"')

    def test_compose_interpolations(self) -> None:
        sites = gen.compose_sites(self.root)
        found = _defaults(sites)
        self.assertNotIn("COMMENTED", found)
        # `$${PORT:-1}` is the container shell's, not compose's.
        self.assertEqual(found["PORT"], ["6555"])
        self.assertEqual(found["ROCR"], ["${GPU:-0}"])
        self.assertEqual(found["GPU"], ["0"])
        by_name = {site.name: site for site in sites}
        self.assertTrue(by_name["PID_MODE"].keeps_empty)
        self.assertFalse(by_name["IMAGE_REF"].keeps_empty)
        self.assertEqual([site.note for site in sites if site.name == "MODEL"], ["server", "emulator", "emulator"])
        self.assertEqual(by_name["PORT"].note, "x-env")
        # Read without a default: listed so it needs an entry, but compose
        # does not become the owner of a default another layer holds.
        self.assertEqual(found["LOGDIR"], [None])
        self.assertEqual(found["MODEL"], ["big", "small", "small"])
        self.assertEqual(gen.owner(gen.collect(self.root)["LOGDIR"]).path, "Makefile")

    def test_makefile_defaults(self) -> None:
        sites = gen.make_sites(self.root, "Makefile")
        self.assertEqual(
            [(site.name, site.default, site.note) for site in sites],
            [
                ("PORT", "6555", ""),
                ("MODEL", "", ""),
                ("LOGDIR", "$(CURDIR)/logs", ""),
                ("LONG", "a b", ""),
                ("PARTITION", "spur-v2", "SPUR"),
                ("LOGDIR", "/shared/logs", "SPUR"),
                ("OTHER", "elsewhere", "conditional"),
            ],
        )

    def test_shell_defaults(self) -> None:
        sites = gen.shell_sites(self.root, gen.SHELL_FILES[0])
        found = _defaults(sites)
        self.assertEqual(found["TIME"], ["04:00:00"])
        self.assertEqual(found["MONITORING"], ["1"])
        self.assertEqual(found["NEEDED"], [None])
        self.assertEqual(found["DRIVER"], ["docker", "docker-container"])
        for name in ("COMMENTED", "LIMIT", "SHOWN"):
            self.assertNotIn(name, found)
        by_name = {site.name: site for site in sites}
        self.assertTrue(by_name["PID_MODE"].keeps_empty)
        self.assertEqual(by_name["NEEDED"].note, "required")
        self.assertEqual([site.note for site in sites if site.name == "DRIVER"], ["", "another branch"])

    def test_python_getenv(self) -> None:
        sites = gen.python_sites(self.root, gen.PYTHON_FILES[0])
        self.assertEqual(
            [(site.name, site.default, site.note) for site in sites],
            [
                ("TEST_URL", "http://server:8000", ""),
                ("TEST_N", "10", ""),
                ("TEST_LIMIT", None, ""),
                ("TEST_DIR", None, "required"),
            ],
        )

    def test_ownership(self) -> None:
        variables = gen.collect(self.root)
        self.assertNotIn("SLURM_JOB_ID", variables)
        # Compose owns what it declares; make and the script only repeat it.
        self.assertEqual(gen.owner(variables["PORT"]).path, str(gen.COMPOSE_FILE))
        self.assertEqual(gen.default_cell(variables["PORT"]), "`6555`")
        # Another service, then the SPUR block: both are different defaults.
        self.assertEqual(gen.default_cell(variables["MODEL"]), "`big`; `small` (`emulator`)")
        self.assertEqual(
            gen.default_cell(variables["LOGDIR"]), "`$(CURDIR)/logs`; `/shared/logs` (SPUR)"
        )
        # A SPUR site default does not own when another file has a default.
        self.assertEqual(gen.owner(variables["PARTITION"]).path, gen.SHELL_FILES[0])
        self.assertEqual(gen.default_cell(variables["PARTITION"]), "`defq`; `spur-v2` (`Makefile`, SPUR)")
        self.assertEqual(gen.default_cell(variables["OTHER"]), "`elsewhere` (conditional)")
        self.assertEqual(
            gen.default_cell(variables["PID_MODE"]),
            f"`service:other` (only when unset); `host` (`{gen.SHELL_FILES[0]}`)",
        )
        self.assertEqual(gen.default_cell(variables["DRIVER"]), "`docker`; `docker-container` (another branch)")
        self.assertEqual(gen.default_cell(variables["NEEDED"]), "(required)")
        self.assertEqual(gen.default_cell(variables["TEST_LIMIT"]), "(unset)")
        # Reading the owner through its helper, or an alias, is not a second default.
        self.assertEqual(gen.default_cell(variables["IMAGE_REF"]), "`server:latest`")
        self.assertTrue(gen.reads_owner("${OLD:-}"))
        self.assertFalse(gen.reads_owner("${OLD:-fallback}"))
        self.assertEqual(gen.default_cell(variables["TAG"]), '`PIN == "" ? _INTERNAL : "${_INTERNAL}-y"`')


class RenderTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = _root(tmp.name, Makefile="PORT ?= 6555\nMODEL ?=\n\ttarget: $(NODE)\n")

    def _write(self, descriptions: str, doc: str = "") -> None:
        (self.root / gen.DESCRIPTIONS).write_text(descriptions, encoding="utf-8")
        (self.root / gen.DOC).write_text(
            doc or "# Doc\n\n<!-- BEGIN generated:stack -->\nstale\n<!-- END generated:stack -->\n\nprose\n",
            encoding="utf-8",
        )

    def _render(self) -> str:
        return gen.render_doc(self.root, (self.root / gen.DOC).read_text(encoding="utf-8"))

    def test_renders_between_the_markers_and_keeps_the_prose(self) -> None:
        self._write("# comment\n[stack]\nPORT: Listen port\n  of the server\nNODE: Pin the node\n[ignore]\nMODEL: internal\n")
        self.assertEqual(
            self._render(),
            "# Doc\n\n<!-- BEGIN generated:stack -->\n"
            "| Variable | Default | Owner | Description |\n"
            "| --- | --- | --- | --- |\n"
            "| `PORT` | `6555` | `Makefile` | Listen port of the server |\n"
            "| `NODE` | (unset) | `Makefile` | Pin the node |\n"
            "<!-- END generated:stack -->\n\nprose\n",
        )
        # Regenerating is a fixed point.
        (self.root / gen.DOC).write_text(self._render(), encoding="utf-8")
        self.assertEqual(gen.check_or_write(self.root, write=False), 0)

    def test_a_variable_in_code_needs_an_entry(self) -> None:
        self._write("[stack]\nPORT: Listen port\n")
        with self.assertRaisesRegex(gen.GenerateError, r"no entry for: MODEL \(Makefile\)"):
            self._render()

    def test_an_entry_needs_a_variable_in_code(self) -> None:
        self._write("[stack]\nPORT: Listen port\nMODEL: Model\nGONE: Removed last year\n")
        with self.assertRaisesRegex(gen.GenerateError, "GONE is described but no longer appears in code"):
            self._render()
        self._write("[stack]\nPORT: Listen port\nMODEL: Model\n[ignore]\nGONE: internal\n")
        with self.assertRaisesRegex(gen.GenerateError, "no longer in code: GONE"):
            self._render()

    def test_sections_and_markers_must_pair_up(self) -> None:
        self._write("[stack]\nPORT: Listen port\nMODEL: Model\n", doc="# Doc\n")
        with self.assertRaisesRegex(gen.GenerateError, "generated:stack"):
            self._render()
        self._write(
            "[stack]\nPORT: Listen port\nMODEL: Model\n",
            doc="<!-- BEGIN generated:stack -->\n<!-- END generated:stack -->\n"
            "<!-- BEGIN generated:other -->\n<!-- END generated:other -->\n",
        )
        with self.assertRaisesRegex(gen.GenerateError, "no section .*: other"):
            self._render()

    def test_description_file_errors(self) -> None:
        for text, message in (
            ("PORT: before any section\n", "before the first"),
            ("[stack]\nPORT: one\n[stack]\n", "appears twice"),
            ("[stack]\nPORT: one\nPORT: two\n", "listed twice"),
            ("[stack]\n  continuation\n", "continuation line"),
            ("[stack]\nnot an entry\n", "expected `NAME: description`"),
            ("[stack]\nPORT: a | b\nMODEL: Model\n", "write '|' as"),
        ):
            with self.subTest(text=text):
                self._write(text)
                with self.assertRaisesRegex(gen.GenerateError, re.escape(message)):
                    self._render()

    def test_check_reports_drift_and_write_fixes_it(self) -> None:
        self._write("[stack]\nPORT: Listen port\nMODEL: Model\n")
        check = [sys.executable, str(SCRIPT), "--root", str(self.root), "--check"]
        result = subprocess.run(check, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("out of sync", result.stderr)
        self.assertIn("+| `PORT` | `6555` | `Makefile` | Listen port |", result.stdout)
        write = [sys.executable, str(SCRIPT), "--root", str(self.root), "--write"]
        self.assertEqual(subprocess.run(write, capture_output=True, check=False).returncode, 0)
        self.assertEqual(subprocess.run(check, capture_output=True, check=False).returncode, 0)


class CommittedReferenceTest(unittest.TestCase):
    def test_reference_is_in_sync(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_list_vars_prints_only_documented_variables(self) -> None:
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        recipe = makefile[makefile.index("\nlist-vars:") :]
        printed = {
            name
            for names in re.findall(r"^\t    '([A-Z][A-Z0-9_ /]*)' '", recipe, re.M)
            for name in names.split(" / ")
        }
        self.assertGreater(len(printed), 50)
        documented = {
            name
            for section, entries in gen.load_descriptions(REPO_ROOT).items()
            if section != gen.IGNORE_SECTION
            for name in entries
        }
        self.assertEqual(sorted(printed - documented), [])

    def test_approved_single_values(self) -> None:
        variables = gen.collect(REPO_ROOT)
        for name, value in (
            ("LMCACHE_MAX_GPU_WORKERS", "`4`"),
            ("LMCACHE_NIXL_POSIX_SLOT_SIZE", "`16777216`"),
            ("RDMA_EXPORTER_VERSION", "`0.7.5`"),
        ):
            with self.subTest(name=name):
                self.assertEqual(gen.default_cell(variables[name]), value)


if __name__ == "__main__":
    unittest.main()

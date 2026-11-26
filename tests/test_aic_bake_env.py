# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""docker/docker-bake.hcl is the single source of build defaults.

Covers docker/scripts/aic-bake-env.sh (the one reader of `bake --print`), the
AIC_VERSION literal bake cannot drop (VERSION owns it), and the Dockerfiles,
which must not hold a default for an arg bake supplies.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BAKE_FILE = REPO_ROOT / "docker" / "docker-bake.hcl"
HELPER = REPO_ROOT / "docker" / "scripts" / "aic-bake-env.sh"
TAG_SCRIPT = REPO_ROOT / "docker" / "scripts" / "aic-image-tag.sh"

IMAGE_REFS = {
    "AIC_BASE_IMAGE_REF": "base",
    "VLLM_IMAGE_REF": "vllm",
    "LMCACHE_IMAGE_REF": "lmcache",
    "HSA_SNOOP_IMAGE_REF": "hsa-snoop",
    "VLLM_EMULATE_IMAGE_REF": "emulate",
    "KVBENCH_IMAGE_REF": "kvbench",
    "NVME_EXPORTER_IMAGE_REF": "nvme-exporter",
    "RDMA_EXPORTER_IMAGE_REF": "rdma-exporter",
    "CLIENT_IMAGE_REF": "client",
}

BAKE_VARIABLES = set(re.findall(r'^variable "([^"]+)"', BAKE_FILE.read_text(encoding="utf-8"), re.M))


def _have_buildx() -> bool:
    try:
        return subprocess.run(
            ["docker", "buildx", "version"], check=False, capture_output=True
        ).returncode == 0
    except OSError:
        return False


def _env(**overrides: str) -> dict[str, str]:
    """The caller's environment without any bake variable, plus overrides."""
    env = {key: value for key, value in os.environ.items() if key not in BAKE_VARIABLES}
    env.update(overrides)
    return env


def _run(*args: str, cwd: Path | None = None, **overrides: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(HELPER), *args],
        cwd=cwd,
        env=_env(**overrides),
        check=False,
        capture_output=True,
        text=True,
    )


def _values(**overrides: str) -> dict[str, str]:
    result = _run("--json", **overrides)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _bake_targets(*targets: str) -> dict[str, dict]:
    result = subprocess.run(
        ["docker", "buildx", "bake", "-f", str(BAKE_FILE), "--print", *targets],
        cwd=REPO_ROOT,
        env=_env(),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["target"]


@unittest.skipUnless(_have_buildx(), "docker buildx is not available")
class AicBakeEnvTest(unittest.TestCase):
    # Exercises quoting in every mode: ';' in the arch list, and "'", '#', '$'
    # and a space in a pin.
    AWKWARD = {"ROCM_ARCH": "gfx942;gfx950", "CLR_REF": "it's #1 $HOME"}

    def test_json_holds_args_tag_and_one_ref_per_target(self) -> None:
        values = _values()
        for name in ("ROCM_VERSION", "VLLM_REF", "LMCACHE_REF", "ROCM_BASE_IMAGE", "IMAGE_TAG"):
            self.assertTrue(values.get(name), name)
        self.assertNotIn("BLAS_CACHE_BUST", values)
        self.assertEqual(values["VLLM_TARGET_DEVICE"], "rocm")
        targets = _bake_targets(*IMAGE_REFS.values())
        for ref, target in IMAGE_REFS.items():
            self.assertEqual(values[ref], targets[target]["tags"][0], ref)
        for target in ("base", "vllm", "lmcache", "emulate"):
            self.assertTrue(values[_ref_of(target)].endswith(f":{values['IMAGE_TAG']}"), target)
        snoop_tag = values["HSA_SNOOP_REF"].removeprefix("v")
        self.assertEqual(values["HSA_SNOOP_IMAGE_REF"], f"aic-hsa-snoop:{snoop_tag}")
        self.assertNotIn(snoop_tag, values["IMAGE_TAG"])

    def test_env_mode_matches_json(self) -> None:
        values = _values(**self.AWKWARD)
        keys = sorted(values)
        result = subprocess.run(
            [
                "bash",
                "-c",
                'eval "$("$0" --env)" && for k in "$@"; do printf "%s\\0" "${!k}"; done',
                str(HELPER),
                *keys,
            ],
            env=_env(**self.AWKWARD),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.split("\0")[:-1], [values[key] for key in keys])

    def test_get_mode_matches_json(self) -> None:
        values = _values(**self.AWKWARD)
        keys = sorted(values)
        result = _run("--get", *keys, **self.AWKWARD)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [values[key] for key in keys])

    def test_get_unknown_name_fails_with_no_output(self) -> None:
        result = _run("--get", "ROCM_VERSION", "NO_SUCH_NAME")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("NO_SUCH_NAME", result.stderr)

    @unittest.skipUnless(shutil.which("make"), "make is not available")
    def test_make_mode_matches_json(self) -> None:
        values = _values(**self.AWKWARD)
        keys = sorted(values)
        fragment = _run("--make", **self.AWKWARD)
        self.assertEqual(fragment.returncode, 0, fragment.stderr)
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "bake.mk").write_text(fragment.stdout, encoding="utf-8")
            (Path(temp) / "Makefile").write_text(
                "include bake.mk\n"
                + "".join(f"$(info {key}=$({key}))\n" for key in keys)
                + "all: ;\n",
                encoding="utf-8",
            )
            # A bare environment: `?=` must not be satisfied by an inherited value.
            result = subprocess.run(
                ["make", "-s", "-C", temp],
                env={"PATH": os.environ["PATH"]},
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.stdout.splitlines(), [f"{key}={values[key]}" for key in keys])

    def test_build_args_are_exactly_the_bake_target_args(self) -> None:
        pinned = {"BLAS_CACHE_BUST": "fixture", **self.AWKWARD}
        for target in ("base", "vllm", "emulate", "wheels-emulate", "kvbench"):
            with self.subTest(target=target):
                result = subprocess.run(
                    [str(HELPER), "--build-args", target],
                    env=_env(**pinned),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                expected = subprocess.run(
                    ["docker", "buildx", "bake", "-f", str(BAKE_FILE), "--print", target],
                    env=_env(**pinned),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                args = json.loads(expected.stdout)["target"][target]["args"]
                self.assertEqual(
                    result.stdout.splitlines(),
                    [f"--build-arg={key}={value}" for key, value in sorted(args.items())],
                )
        base = _run("--build-args", "base").stdout
        self.assertIn("--build-arg=BLAS_CACHE_BUST=\n", base)
        emulate = _run("--build-args", "emulate").stdout
        self.assertIn("--build-arg=VLLM_TARGET_DEVICE=empty\n", emulate)

    def test_build_args_unknown_target_fails(self) -> None:
        result = _run("--build-args", "no-such-target")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("no-such-target", result.stderr)

    def test_environment_overrides_flow_through(self) -> None:
        values = _values(
            VLLM_REF="v9.9.9",
            ROCM_ARCH="gfx942,gfx950",
            ROCM_VERSION="9.8.7",
            KVBENCH_REF="0123456789abcdef",
            AIC_IMAGE_PREFIX="registry.example:5000/aic/",
            BUILD_JOBS="",
        )
        self.assertEqual(values["VLLM_REF"], "v9.9.9")
        self.assertEqual(values["ROCM_ARCH"], "gfx942;gfx950")
        self.assertEqual(values["ROCM_BASE_IMAGE"], "rocm/dev-ubuntu-26.04:9.8.7-full")
        self.assertEqual(values["BUILD_JOBS"], "")
        self.assertIn("-rocm9.8.7-", values["IMAGE_TAG"])
        self.assertIn("-vllm9.9.9-", values["IMAGE_TAG"])
        self.assertTrue(values["IMAGE_TAG"].endswith("-gfx942-950"))
        self.assertNotIn("/", values["IMAGE_TAG"])
        self.assertEqual(
            values["VLLM_IMAGE_REF"], f"registry.example:5000/aic/aic-vllm:{values['IMAGE_TAG']}"
        )
        self.assertEqual(values["KVBENCH_IMAGE_REF"], "registry.example:5000/aic/aic-kvbench:0123456")

    def test_image_tag_equals_tag_script(self) -> None:
        for arch in ("", "gfx1201", "gfx942;gfx950"):
            with self.subTest(arch=arch):
                tag = subprocess.run(
                    ["bash", str(TAG_SCRIPT)],
                    env=_env(ROCM_ARCH=arch),
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.assertEqual(tag, _values(ROCM_ARCH=arch)["IMAGE_TAG"] + "\n")

    def test_works_from_any_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            result = _run("--json", cwd=Path(temp))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), _values())

    def test_aic_version_comes_from_the_version_file(self) -> None:
        version = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
        self.assertEqual(_values()["AIC_VERSION"], version)
        self.assertEqual(_values(AIC_VERSION="")["AIC_VERSION"], version)
        self.assertEqual(_values(AIC_VERSION="9.9.9")["AIC_VERSION"], "9.9.9")

    def test_variables_are_the_hcl_variable_names(self) -> None:
        result = _run("--variables")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()), BAKE_VARIABLES)
        # Without docker buildx the helper reads the names off the HCL instead.
        with tempfile.TemporaryDirectory() as temp:
            docker = Path(temp) / "docker"
            docker.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            docker.chmod(0o755)
            fallback = _run("--variables", PATH=f"{temp}{os.pathsep}{os.environ['PATH']}")
        self.assertEqual(fallback.returncode, 0, fallback.stderr)
        self.assertEqual(fallback.stdout, result.stdout)
        # Likewise with a buildx whose bake has no --list.
        with tempfile.TemporaryDirectory() as temp:
            docker = Path(temp) / "docker"
            docker.write_text(
                '#!/bin/sh\n[ "$2" = version ] && exit 0\necho fixture-no-list >&2\nexit 1\n',
                encoding="utf-8",
            )
            docker.chmod(0o755)
            old_bake = _run("--variables", PATH=f"{temp}{os.pathsep}{os.environ['PATH']}")
        self.assertEqual(old_bake.returncode, 0, old_bake.stderr)
        self.assertEqual(old_bake.stdout, result.stdout)
        self.assertIn("fixture-no-list", old_bake.stderr)

    def test_bake_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            docker = Path(temp) / "docker"
            docker.write_text("#!/bin/sh\necho fixture-bake-error >&2\nexit 3\n", encoding="utf-8")
            docker.chmod(0o755)
            result = _run("--json", PATH=f"{temp}{os.pathsep}{os.environ['PATH']}")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("fixture-bake-error", result.stderr)


def _ref_of(target: str) -> str:
    return next(ref for ref, name in IMAGE_REFS.items() if name == target)


class AicVersionLiteralTest(unittest.TestCase):
    def test_hcl_literal_equals_version_file(self) -> None:
        literal = re.findall(
            r'^variable "AIC_VERSION"\s*\{\s*default = "([^"]*)"\s*\}',
            BAKE_FILE.read_text(encoding="utf-8"),
            re.M,
        )
        version = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
        self.assertEqual(literal, [version], "docker/docker-bake.hcl AIC_VERSION must equal VERSION")


ARG_LINE = re.compile(r"^[ \t]*ARG[ \t]+([A-Za-z_][A-Za-z0-9_]*)(?:=(.*?))?[ \t]*$", re.M)

# Compose also builds these images with args of its own, so their ARG defaults
# stay; they must equal what bake passes.  Keyed by bake target.
KEEPS_DEFAULTS = {
    "kvbench": "docker/kvbench/Dockerfile",
    "client": "docker/client/Dockerfile",
}


def _dockerfiles() -> list[Path]:
    docker = REPO_ROOT / "docker"
    return sorted({*docker.rglob("Dockerfile"), *docker.rglob("*.Dockerfile")})


def _args(dockerfile: Path) -> list[tuple[str, str | None]]:
    """(name, default) per ARG line; default is None for a bare `ARG NAME`."""
    return [
        (match.group(1), match.group(2))
        for match in ARG_LINE.finditer(dockerfile.read_text(encoding="utf-8"))
    ]


class DockerfileArgOwnershipTest(unittest.TestCase):
    """A Dockerfile `ARG X=<default>` would be a second copy of a bake default."""

    def test_no_dockerfile_defaults_a_bake_supplied_arg(self) -> None:
        hcl = BAKE_FILE.read_text(encoding="utf-8")
        # Every variable, plus every key of a target's `args` block.
        supplied = BAKE_VARIABLES | set(re.findall(r"^[ \t]+([A-Z][A-Z0-9_]*)[ \t]*=", hcl, re.M))
        self.assertIn("ROCM_APT_FROM", supplied)
        dockerfiles = _dockerfiles()
        self.assertGreaterEqual(len(dockerfiles), 11)
        problems = [
            f"{dockerfile.relative_to(REPO_ROOT)}: ARG {name}={default}"
            for dockerfile in dockerfiles
            if dockerfile.relative_to(REPO_ROOT).as_posix() not in KEEPS_DEFAULTS.values()
            for name, default in _args(dockerfile)
            if default is not None and name in supplied
        ]
        self.assertEqual(
            problems, [], "docker/docker-bake.hcl owns these; declare them as a bare `ARG NAME`"
        )

    @unittest.skipUnless(_have_buildx(), "docker buildx is not available")
    def test_dockerfiles_declare_every_arg_bake_passes(self) -> None:
        targets = _bake_targets(
            "all", "wheels", "emulate-wheels", "emulate", "client", "lmcache-wheels", "from-wheels"
        )
        problems = []
        for name, target in sorted(targets.items()):
            declared = {arg for arg, _ in _args(REPO_ROOT / target["dockerfile"])}
            for arg in sorted(set(target.get("args", {})) - declared):
                problems.append(f"bake target {name} passes {arg}; {target['dockerfile']} has no such ARG")
        self.assertEqual(problems, [])

    @unittest.skipUnless(_have_buildx(), "docker buildx is not available")
    def test_bake_passes_every_arg_a_dockerfile_guards(self) -> None:
        targets = _bake_targets(
            "all", "wheels", "emulate-wheels", "emulate", "client", "lmcache-wheels", "from-wheels"
        )
        problems = []
        for name, target in sorted(targets.items()):
            text = (REPO_ROOT / target["dockerfile"]).read_text(encoding="utf-8")
            guarded = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):\?unset: build through", text))
            for arg in sorted(guarded - set(target.get("args", {}))):
                problems.append(f"{target['dockerfile']} guards {arg}; bake target {name} does not pass it")
        self.assertEqual(problems, [])

    @unittest.skipUnless(_have_buildx(), "docker buildx is not available")
    def test_kept_defaults_equal_bake(self) -> None:
        compose = (REPO_ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8")
        targets = _bake_targets(*KEEPS_DEFAULTS)
        for name, dockerfile in KEEPS_DEFAULTS.items():
            with self.subTest(target=name):
                args = targets[name]["args"]
                self.assertEqual(dict(_args(REPO_ROOT / dockerfile)), args)
                # A compose build passes its own, which override the ARG defaults.
                build = re.search(
                    rf"^ {{4}}build:\n {{6}}context: \.\.\n {{6}}dockerfile: {re.escape(dockerfile)}\n"
                    r" {6}args:\n((?: {8}\S.*\n)+)",
                    compose,
                    re.M,
                )
                self.assertIsNotNone(build, f"compose has no build of {dockerfile}")
                passed = dict(re.findall(r"^ {8}(\w+): \$\{\1:-([^}]*)\}$", build.group(1), re.M))
                self.assertEqual(passed, args)


@unittest.skipUnless(_have_buildx(), "docker buildx is not available")
class ComposeImageDefaultsTest(unittest.TestCase):
    """Compose's fallback for a locally built image must be a tag bake applies."""

    COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.yml"
    LEGACY = ("IMAGE_REF", "IMAGE_NAME")

    def test_defaults_name_images_bake_tags(self) -> None:
        tags = {tag for target in _bake_targets("all", "client").values() for tag in target.get("tags", [])}
        text = self.COMPOSE_FILE.read_text(encoding="utf-8")
        defaults = re.findall(r"^\s*image:\s*\$\{(\w+):-(aic-[^}]+)\}\s*$", text, re.M)
        self.assertGreaterEqual(len(defaults), 9)
        wrong = sorted({f"{var} -> {image}" for var, image in defaults if image not in tags})
        self.assertEqual(wrong, [], "compose defaults bake does not tag")

    def test_no_single_image_tier(self) -> None:
        images = re.findall(r"^\s*image:\s*(.+)$", self.COMPOSE_FILE.read_text(encoding="utf-8"), re.M)
        legacy = [line for line in images if any("${" + name in line for name in self.LEGACY)]
        self.assertEqual(legacy, [])


if __name__ == "__main__":
    unittest.main()

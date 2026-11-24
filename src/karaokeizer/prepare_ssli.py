"""Build a pinned SSLI distribution with sample-accurate formant scheduling."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

PINNED_SSLI_COMMIT = "882a1e061d75c3db95b6e63fd8ab9980102e24d4"
CAPABILITY_SCHEMA = "karaokeizer-ssli-capabilities/v1"
CAPABILITY_FILE = "karaokeizer-ssli-capabilities.json"


def run(command: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a bounded build command without invoking a shell."""

    return subprocess.run(command, cwd=cwd, check=check, text=True, capture_output=True)


def sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source: Path, output: Path) -> Path:
    """Patch a pristine pinned SSLI checkout, build it, then restore the checkout."""

    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be new or empty: {output}")
    commit = run(["git", "rev-parse", "HEAD"], source).stdout.strip()
    if commit != PINNED_SSLI_COMMIT:
        raise ValueError(f"SSLI checkout must be pinned to {PINNED_SSLI_COMMIT}; found {commit}")
    if run(["git", "status", "--porcelain", "--untracked-files=no"], source).stdout.strip():
        raise ValueError("SSLI checkout has tracked modifications; use a pristine checkout")
    patch = Path(str(files("karaokeizer").joinpath("data/ssli-formant-scheduler.patch")))
    run(["git", "apply", "--check", str(patch)], source)
    output.mkdir(parents=True, exist_ok=True)
    applied = False
    try:
        run(["git", "apply", str(patch)], source)
        applied = True
        build = run(
            [sys.executable, "build.py", "ssli", "--output", str(output / "index.html"), "--no-version-bump"],
            source,
        )
        if not (output / "index.html").is_file():
            raise RuntimeError(f"SSLI build did not produce index.html: {build.stderr}")
        shutil.copytree(source / "assets", output / "assets")
        for worklet in (source / "shared" / "assets" / "worklets").glob("*.js"):
            shutil.copy2(worklet, output / "assets" / worklet.name)
        for original, destination in (("LICENSE", "SSLI-LICENSE.txt"), ("NOTICE", "SSLI-NOTICE.txt")):
            shutil.copy2(source / original, output / destination)
        capability = {
            "schema": CAPABILITY_SCHEMA,
            "upstream_commit": commit,
            "formant_scheduling_api": 1,
            "patch_sha256": sha256(patch),
            "index_sha256": sha256(output / "index.html"),
        }
        (output / CAPABILITY_FILE).write_text(json.dumps(capability, indent=2) + "\n", encoding="utf-8")
        return output
    finally:
        if applied:
            run(["git", "apply", "--reverse", str(patch)], source, check=False)


def main() -> int:
    """CLI entry point with concise operator-facing validation errors."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pristine checkout of the pinned SuperSynthLabInstrument commit")
    parser.add_argument("output", type=Path, help="New or empty distribution directory")
    args = parser.parse_args()
    try:
        destination = prepare(args.source, args.output)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

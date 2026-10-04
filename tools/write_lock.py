"""Regenerate backend/requirements.lock from the built API image.

The freeze has to happen inside the image, not on a developer machine: the
deploy target is python:3.12-slim, and a freeze taken on any other interpreter
pins versions that either do not exist there or, in tensorflow-cpu's case, have
no wheel at all for 3.13+.

    make lock          # builds the image first, then runs this
    .\\make.ps1 lock
"""

import subprocess
import sys
from pathlib import Path

IMAGE = "ev-fleet-intelligence-api:latest"
LOCK = Path(__file__).resolve().parents[1] / "backend" / "requirements.lock"
SKIP = ("pip==", "setuptools==", "wheel==")

HEADER = """\
# Exact dependency set the API image was built and verified against.
#
# Generated on python:3.12-slim (the deploy target) with `make lock`, which
# freezes the resolved tree from inside the built image rather than from a
# developer machine -- a freeze taken on a different interpreter would pin
# versions that do not exist for 3.12, and tensorflow-cpu has no wheel at all
# for 3.13+.
#
# requirements.txt holds the compatibility envelope and is what the Dockerfile
# installs. This file records what a release actually ran, so a result can be
# reproduced months later after upstream has moved. Regenerate with:
#
#     make lock          (or:  .\\make.ps1 lock)
#
"""


def main() -> int:
    try:
        out = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "sh", IMAGE,
             "-c", "pip freeze --exclude-editable"],
            capture_output=True, text=True, check=True, timeout=300).stdout
    except FileNotFoundError:
        print("docker not found on PATH", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"docker run failed: {exc.stderr.strip()[:400]}", file=sys.stderr)
        print(f"build the image first:  docker build -t {IMAGE} .", file=sys.stderr)
        return 1

    pins = sorted(
        line.strip() for line in out.splitlines()
        if "==" in line and not line.strip().lower().startswith(SKIP)
    )
    if not pins:
        print("pip freeze returned nothing -- is the image built?", file=sys.stderr)
        return 1

    LOCK.write_text(HEADER + "\n".join(pins) + "\n", encoding="utf-8")
    print(f"wrote {LOCK.relative_to(LOCK.parents[1])} with {len(pins)} pinned packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

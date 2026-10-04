"""Artifact inventory behind /health and /ready.

Liveness and readiness are separate on purpose:
  /health  the process is up. Never touches disk, so it is safe to poll often.
  /ready   the artifacts and libraries each agent needs are actually usable,
           with a sha256 so you can tell WHICH build is loaded.

Availability is measured from disk on every call. A hardcoded agent list makes a
deploy with no model artifacts look healthy from every angle.
"""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from app import config

PROJECT_ROOT = config.PROJECT_ROOT
MODELS_DIR = config.MODELS_DIR
PROCESSED_DIR = config.PROCESSED_DATA_DIR


@dataclass(frozen=True)
class Artifact:
    name: str
    path: Path
    agents: tuple
    required: bool = True


ARTIFACTS = (
    Artifact("battery_rul_model", MODELS_DIR / "battery_rul_model.pkl", ("battery",)),
    Artifact("quality_casting", MODELS_DIR / "quality_model.keras", ("quality",)),
    Artifact("quality_neu_det", MODELS_DIR / "quality_model_neu_det.keras", ("quality",)),
    Artifact("risk_model", MODELS_DIR / "risk_model.pkl", ("risk",)),
    Artifact("shipments_csv", PROCESSED_DIR / "supply_chain_shipments.csv",
             ("risk", "compound_risk")),
    Artifact("battery_features_csv", PROCESSED_DIR / "battery_features.csv",
             ("battery_live_feed",)),
)

# agents that are pure logic and need no artifact at all
ALWAYS_AVAILABLE = ("carbon", "fleet_readiness", "maintenance")

# An agent can also be unusable because a LIBRARY is missing, not a file. Two
# 98 MB .keras artifacts on disk say nothing about whether this interpreter can
# load them: tensorflow-cpu has no wheel for Python 3.13+ or arm64 Linux.
# Reporting `quality` as available in that case is the same "200 while broken"
# shape a hardcoded agent list produces, one layer down.
RUNTIME_DEPENDENCIES = (
    ("tensorflow", ("quality",)),
)


def _missing_runtime_deps() -> dict:
    from app.agents.quality_agent import TENSORFLOW_IMPORT_ERROR

    errors = {"tensorflow": TENSORFLOW_IMPORT_ERROR}
    return {name: {"agents": list(agents), "reason": str(errors[name])}
            for name, agents in RUNTIME_DEPENDENCIES
            if errors.get(name) is not None}

def _display_path(path: Path) -> str:
    """Repo-relative where possible, bare filename otherwise.

    EV_FLEET_MODELS_DIR can point anywhere, and `relative_to` raises when it
    points outside the repo. Falls back to the filename rather than the absolute
    path, which /ready returns to unauthenticated callers.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


@lru_cache(maxsize=32)
def _sha256_cached(path_str: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with Path(path_str).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    """Cached by (path, mtime, size) -- hashing a 98 MB .keras on every poll
    would make the readiness probe the most expensive endpoint on the service.

    Bounded: mtime is part of the key, so a plain dict grew by one permanent
    entry on every retrain of every artifact. lru_cache evicts instead.
    """
    stat = path.stat()
    return _sha256_cached(str(path), stat.st_mtime_ns, stat.st_size)


def inventory(include_hashes: bool = True) -> dict:
    artifacts, missing = {}, []
    for art in ARTIFACTS:
        present = art.path.is_file()
        entry = {"present": present, "required": art.required,
                 "agents": list(art.agents)}
        if present:
            stat = art.path.stat()
            entry["size_bytes"] = stat.st_size
            entry["modified_utc"] = datetime.fromtimestamp(
                stat.st_mtime, tz=UTC).isoformat(timespec="seconds")
            if include_hashes:
                entry["sha256"] = _sha256(art.path)[:16]
        else:
            entry["expected_at"] = _display_path(art.path)
            missing.append(art.name)
        artifacts[art.name] = entry

    missing_deps = _missing_runtime_deps()
    degraded = {a for art in ARTIFACTS if not art.path.is_file() for a in art.agents}
    degraded |= {a for dep in missing_deps.values() for a in dep["agents"]}
    available = [a for a in
                 ("battery", "quality", "risk", "fleet_readiness",
                  "compound_risk", "carbon", "maintenance")
                 if a not in degraded]

    return {
        # a missing library is as disqualifying as a missing file -- both mean
        # an agent listed in the docs cannot answer
        "ready": not missing and not missing_deps,
        "artifacts": artifacts,
        "missing": missing,
        "missing_runtime_dependencies": missing_deps,
        # measured, not asserted -- this is the field the old root endpoint
        # hardcoded to "all seven"
        "agents_available": available,
        "agents_degraded": sorted(degraded),
        "agents_needing_no_artifact": list(ALWAYS_AVAILABLE),
    }

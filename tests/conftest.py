"""Shared fixtures: synthetic cases, and the real sample frames when they are present."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sunmosaic.synthetic import quadrant_case

# The regression values are measured from the four frames in demo/, so the tests read those by
# default rather than a working folder that may hold other frames.
SAMPLE_DIR = Path(os.environ.get("SUNMOSAIC_SAMPLES", Path(__file__).resolve().parents[1] / "demo"))


@pytest.fixture(scope="session")
def sample_paths() -> list[str]:
    """The four real H-alpha frames from demo/; the test is skipped when they are missing."""
    if not SAMPLE_DIR.is_dir():
        pytest.skip(f"sample frames not found in {SAMPLE_DIR}")
    paths = sorted(str(p) for p in SAMPLE_DIR.glob("*.tif"))
    if len(paths) < 2:
        pytest.skip(f"fewer than two sample frames in {SAMPLE_DIR}")
    return paths


@pytest.fixture(scope="session")
def quad_case():
    """Four synthetic quadrant tiles with known offsets and gains."""
    return quadrant_case(seed=7)


@pytest.fixture(scope="session")
def flat_case():
    """Four synthetic tiles with pure gain differences and no illumination gradient."""
    return quadrant_case(seed=11, gains=(1.0, 0.93, 1.05, 0.98, 1.02), jitter=True)


@pytest.fixture(scope="session")
def center_case():
    """Four quadrant tiles plus a fifth that lies wholly inside the disk (no limb)."""
    return quadrant_case(seed=5, with_center=True, center_size=(900, 700))

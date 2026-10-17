"""Load the paper author's published FooPlayer without editing its source.

Run fetch_published.py to download the author's source at the pinned commit,
or set HEXMACHINA_SOURCE_ROOT to an equivalent checkout. An adapter shim maps
the published interface to our pinned Catanatron. Source is not copied into
this repository.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import random
import sys
import types

from catanLabs.FromScratch.StrategyOne.experiments.hexmachina_web import published_adapter

SOURCE_RELATIVE = Path("supplementary/Evolutions/GPT Evolution 2/foo_player.py")
SOURCE_ROOT = Path(os.environ.get("HEXMACHINA_SOURCE_ROOT", str(Path(__file__).resolve().parent / ".source_cache")))
PUBLISHED_SOURCE = SOURCE_ROOT / SOURCE_RELATIVE


def load_published_class():
    if not PUBLISHED_SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing {PUBLISHED_SOURCE}; clone the author's repository at the pinned commit"
        )
    package_name = "hexmachina_published_exact"
    if package_name + ".foo_player" in sys.modules:
        return sys.modules[package_name + ".foo_player"].FooPlayer
    package = types.ModuleType(package_name)
    package.__path__ = [str(PUBLISHED_SOURCE.parent)]
    sys.modules[package_name] = package
    sys.modules[package_name + ".adapters"] = published_adapter
    spec = importlib.util.spec_from_file_location(package_name + ".foo_player", PUBLISHED_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {PUBLISHED_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FooPlayer


def create_agent(color, checkpoint=None, seed=0):
    # Published FooPlayer hardcodes BLUE in __init__. Assign actual seat after
    # construction; the policy code itself remains untouched.
    random.seed(seed * 2 + (0 if color.value == "RED" else 1))
    player = load_published_class()()
    player.color = color
    return player

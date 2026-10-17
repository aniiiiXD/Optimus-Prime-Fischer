"""Build Eli's native environment outside the immutable source checkouts.

Uses the one workspace environment. Copies source into ignored build artifacts,
and retains the resolved Cargo lockfile for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parent


def build_binding(name: str):
    checkout = ROOT / "vendor" / name
    expected = json.loads((ROOT / "sources.json").read_text())[name]["commit"]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Refusing an unpinned source revision: {checkout}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout, text=True).strip():
        raise RuntimeError(f"Refusing modified source: {checkout}")
    build = ROOT / "artifacts/build" / name
    if name == "eli":
        source = ROOT / "vendor/eli/rust"
        for crate in ("catan-core", "catan-env", "catan-py"):
            shutil.copytree(source / crate, build / crate, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("target", "__pycache__"))
        crate_root, library_name = build / "catan-py", "catan_py"
    else:
        source = ROOT / "vendor/benjamin"
        shutil.copytree(source / "crates/catan_engine", build / "crates/catan_engine",
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns("target"))
        for file in ("Cargo.toml", "Cargo.lock"):
            shutil.copy2(source / file, build / file)
        crate_root, library_name = build, "catan_engine"
    lock = ROOT / "locks" / f"{name}-binding.Cargo.lock"
    if lock.exists():
        shutil.copy2(lock, crate_root / "Cargo.lock")
    env = dict(os.environ, PYO3_PYTHON=sys.executable,
               CARGO_TARGET_DIR=str(ROOT / "artifacts/rust-eli"))
    if sys.platform == "darwin":
        # PyO3 extension modules resolve Python symbols in the importing process.
        # Plain cargo needs the same macOS link flags maturin normally supplies.
        env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") +
                            " -C link-arg=-undefined -C link-arg=dynamic_lookup").strip()
    command = ["cargo", "build", "--release", "--jobs", "2", "--manifest-path",
               str(crate_root / "Cargo.toml")]
    if name == "benjamin":
        command += ["--features", "extension-module"]
    if lock.exists():
        command.append("--locked")
    started = time.perf_counter()
    subprocess.run(command, env=env, check=True)
    native = ROOT / "artifacts/native"
    native.mkdir(parents=True, exist_ok=True)
    library = ROOT / "artifacts/rust-eli/release" / (
        f"lib{library_name}.dylib" if sys.platform == "darwin" else f"lib{library_name}.so")
    extension = native / (library_name + sysconfig.get_config_var("EXT_SUFFIX"))
    shutil.copy2(library, extension)
    lock.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(crate_root / "Cargo.lock", lock)
    record = {"build_s": time.perf_counter()-started, "python": sys.version,
              "extension": str(extension.relative_to(ROOT)),
              "source_commit": json.loads((ROOT / "sources.json").read_text())[name]["commit"],
              "cargo_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
              "rustflags": env.get("RUSTFLAGS", ""),
              "command": command}
    (native / f"build_{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", nargs="+", choices=["eli", "benjamin"], default=["eli", "benjamin"])
    for name in parser.parse_args().agents:
        build_binding(name)


if __name__ == "__main__":
    main()

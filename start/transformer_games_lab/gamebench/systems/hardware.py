"""Conservative CPU capability discovery for reproducible reports."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any


def _command(*args: str) -> str | None:
    try:
        completed = subprocess.run(args, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _sysctl(name: str) -> str | None:
    return _command("sysctl", "-n", name)


def cpu_capabilities() -> dict[str, Any]:
    machine = platform.machine().lower()
    result: dict[str, Any] = {
        "architecture": machine,
        "cpu_name": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "simd": {},
        "matrix": {},
        "library_backend": {},
    }
    if platform.system() == "Darwin":
        result["cpu_name"] = _sysctl("machdep.cpu.brand_string") or result["cpu_name"]
        if machine in {"arm64", "aarch64"}:
            result["simd"] = {
                "neon_128": _sysctl("hw.optional.neon") == "1",
                "fp16": _sysctl("hw.optional.arm.FEAT_FP16") == "1",
                "bf16": _sysctl("hw.optional.arm.FEAT_BF16") == "1",
                "dot_product": _sysctl("hw.optional.arm.FEAT_DotProd") == "1",
                "i8mm": _sysctl("hw.optional.arm.FEAT_I8MM") == "1",
            }
            result["matrix"] = {
                "arm_sme": _sysctl("hw.optional.arm.FEAT_SME") == "1",
                "arm_sme2": _sysctl("hw.optional.arm.FEAT_SME2") == "1",
                "arm_sme_bf16": _sysctl("hw.optional.arm.FEAT_SME_B16B16") == "1",
                "apple_amx_directly_detectable": False,
            }
        else:
            features = " ".join(filter(None, [
                _sysctl("machdep.cpu.features"),
                _sysctl("machdep.cpu.leaf7_features"),
            ])).upper()
            result["simd"] = {
                "sse4_2": "SSE4.2" in features,
                "avx": "AVX1.0" in features or "AVX " in features,
                "avx2": "AVX2" in features,
                "avx512": "AVX512" in features,
            }
            result["matrix"] = {"intel_amx": "AMX" in features}
    elif platform.system() == "Linux":
        cpuinfo = ""
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                cpuinfo = handle.read().lower()
        except OSError:
            pass
        result["simd"] = {
            "neon_128": "neon" in cpuinfo or "asimd" in cpuinfo,
            "avx": " avx " in f" {cpuinfo} ",
            "avx2": "avx2" in cpuinfo,
            "avx512": "avx512" in cpuinfo,
        }
        result["matrix"] = {
            "arm_sme": " sme " in f" {cpuinfo} ",
            "intel_amx": "amx_" in cpuinfo,
        }
    try:
        import torch

        config = torch.__config__.show()
        result["library_backend"] = {
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "accelerate": "BLAS_INFO=accelerate" in config,
            "mkl": "USE_MKL=ON" in config,
            "mkldnn": "USE_MKLDNN=ON" in config,
            "xnnpack": "USE_XNNPACK" in config,
            "reported_cpu_capability": next(
                (line.split(":", 1)[1].strip() for line in config.splitlines()
                 if "CPU capability usage:" in line),
                None,
            ),
        }
    except ImportError:
        result["library_backend"] = {"torch": None}
    return result

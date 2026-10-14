"""ctypes wrapper for the native scalar/SIMD dot-product kernels."""

from __future__ import annotations

from array import array
import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import subprocess
from typing import Any, Literal


Kernel = Literal["scalar", "vector"]


def build_native_library(project_root: Path) -> Path:
    source = project_root / "native" / "simd_kernels.c"
    build = project_root / "build"
    build.mkdir(parents=True, exist_ok=True)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    output = build / f"libgamebench_simd{suffix}"
    if output.exists() and output.stat().st_mtime >= source.stat().st_mtime:
        return output
    linker = "-dynamiclib" if platform.system() == "Darwin" else "-shared"
    base = ["clang", "-O3", "-std=c11", linker, "-fPIC", str(source), "-o", str(output)]
    native = [*base[:3], "-march=native", *base[3:]]
    try:
        subprocess.run(native, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run(base, check=True)
    return output


@dataclass(frozen=True)
class NativeDotConfig:
    elements: int = 1_048_576
    kernel: Kernel = "vector"

    def __post_init__(self) -> None:
        if self.elements <= 0:
            raise ValueError("elements must be positive")
        if self.kernel not in {"scalar", "vector"}:
            raise ValueError("kernel must be scalar or vector")


class NativeDotWorkload:
    def __init__(self, config: NativeDotConfig, library_path: Path) -> None:
        self.config = config
        self.library = ctypes.CDLL(str(library_path))
        signature = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self.function = getattr(self.library, f"dot_{config.kernel}")
        self.function.argtypes = signature
        self.function.restype = ctypes.c_float
        self.library.dot_reference.argtypes = signature
        self.library.dot_reference.restype = ctypes.c_double
        self.library.vector_backend.restype = ctypes.c_char_p
        self.backend = self.library.vector_backend().decode("utf-8")
        self.left = array("f", ((index % 251) / 251.0 for index in range(config.elements)))
        self.right = array("f", ((index % 127) / 127.0 for index in range(config.elements)))
        self.left_ptr = (ctypes.c_float * config.elements).from_buffer(self.left)
        self.right_ptr = (ctypes.c_float * config.elements).from_buffer(self.right)
        self.reference_result = self.library.dot_reference(
            self.left_ptr, self.right_ptr, self.config.elements
        )
        self.name = f"native_dot/{config.kernel}/n{config.elements}"
        self._last = 0.0

    def run(self) -> None:
        self._last = self.function(self.left_ptr, self.right_ptr, self.config.elements)

    def synchronize(self) -> None:
        pass

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self.config),
            "compiled_vector_backend": self.backend,
            "result": self._last,
            "float64_accumulation_reference": self.reference_result,
            "relative_error_vs_reference": abs(self._last - self.reference_result)
            / max(abs(self.reference_result), 1e-30),
            "scalar_vectorization": "explicitly disabled",
        }

    def work_items_per_call(self) -> int:
        return self.config.elements

    def estimated_flops_per_call(self) -> float:
        return float(2 * self.config.elements)

    def estimated_bytes_per_call(self) -> float:
        return float(2 * self.config.elements * 4)

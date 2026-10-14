# SIMD and Matrix Acceleration Lab

## First: separate the names

**SIMD** is the general idea: one instruction performs the same operation on
several numbers. The exact instruction family depends on the CPU:

| CPU family | Vector ISA | float32 values per common vector |
|---|---|---:|
| Arm/Apple Silicon | NEON / Advanced SIMD | 4 in 128 bits |
| x86-64 | AVX2 | 8 in 256 bits |
| x86-64 | AVX-512 | 16 in 512 bits |

**AVX** is therefore an x86 implementation of SIMD, not a feature Apple
Silicon should have. `NO AVX` is correct on an Arm Mac.

**AMX** is ambiguous. Intel AMX means Intel's tile matrix extensions. “Apple
AMX” is a community name for Apple matrix hardware and is not a portable public
instruction contract. This Apple M5 also exposes the architected Arm **SME and
SME2** matrix extensions. Libraries may use matrix hardware internally, but a
feature bit alone does not prove which kernel ran.

## Where these appear in a Transformer

```text
token embedding lookup          mostly memory/indexing
Q, K, V projections             matrix multiplication
Q @ Kᵀ attention scores         matrix multiplication
softmax                         vector math + reductions
attention probabilities @ V     matrix multiplication
MLP linear layers               matrix multiplication
GELU and LayerNorm              vector math + reductions
policy head                     matrix multiplication
```

Large linear layers are candidates for matrix kernels. Elementwise operations,
normalization, and reductions are SIMD candidates. Tiny shapes may lose more
time in dispatch and threading than they gain from wider hardware.

## The three questions we measure separately

1. **Can the CPU execute it?** `hardware_report.py` reads conservative feature
   flags and records the installed PyTorch backend.
2. **Did our native loop vectorize?** `simd_benchmark.py` compiles one scalar
   loop with vectorization disabled and one explicit NEON/AVX2 loop. On macOS,
   `otool -tvV build/libgamebench_simd.dylib` can confirm instructions such as
   `fmla.4s` in `dot_vector`.
3. **Does the ML library have a fast kernel?** `matrix_benchmark.py` measures
   PyTorch GEMM across shape, dtype, and requested thread count. It does not
   claim which private Accelerate implementation was selected.

## Commands

```bash
../.venv/bin/python hardware_report.py

../.venv/bin/python simd_benchmark.py \
  --elements 1048576 \
  --output reports/simd.json

../.venv/bin/python matrix_benchmark.py \
  --sizes 128,512,1024,2048 \
  --dtypes float32,bfloat16 \
  --threads 1,4 \
  --output reports/matrix.json
```

The SIMD dot product has low arithmetic intensity: two FLOPs for eight input
bytes, or about **0.25 FLOP/byte** before cache effects. It normally becomes
bandwidth-bound once the vectors exceed cache.

A square matrix multiply performs approximately `2N³` FLOPs while its minimum
tensor traffic grows like `3N² × element_size`. Its idealized arithmetic
intensity therefore grows with `N`, which is why sufficiently large GEMMs are
usually compute-bound.

## How to avoid lying to ourselves

- Compare numerical outputs, not only runtime. Different accumulation orders
  produce small floating-point differences.
- Benchmark multiple sizes. Small matrices can live entirely in cache.
- Keep dtype explicit. Hardware support does not guarantee an optimized kernel
  in a specific framework build.
- Treat requested PyTorch threads separately from threads used inside BLAS or
  Accelerate; the backend may manage its own worker pool.
- Use medians and tail latency, and rerun in fresh processes.
- Arithmetic intensity based on tensor sizes is a lower-bound traffic model,
  not measured DRAM traffic.
- To prove exact library kernel dispatch, use disassembly, Instruments, or
  hardware counters. Timing alone demonstrates performance, not provenance.

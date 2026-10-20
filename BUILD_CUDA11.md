# CUDA 11.6 / GCC 9 build

The project now uses C++17. The two host-side `std::bit_width` calls have an
equivalent integer implementation, including the zero and uint64 boundary cases.
CUDA AES, PRP, insertion, seed derivation, journal format and pause/resume behavior
are unchanged.

For CUDA 11.6 on RTX 4090, use a new build directory:

```bash
cmake -S . -B build-cuda11 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda11 -j4
ctest --test-dir build-cuda11 --output-on-failure
python3 -m pip install cryptography
python3 smoke_test.py --exe ./build-cuda11/cuckoo_bench --out ./smoke_cuda11_01 --require-failures
```

CUDA 11.6 cannot generate native Ada `sm_89` code. CMake's `86` target emits both
Ampere machine code and compute_86 PTX for compatibility with Ada. Do not use
`native` or `89` with this toolkit on RTX 4090. With CUDA 11.8 or newer, `native`
may be used instead. CMake 3.24 or newer is still required.

The first real GPU launch may include JIT overhead. A successful build is not a
CPU/GPU correctness result; the smoke report must say `passed`.

AES-NI is an x86 CPU instruction set and cannot run inside CUDA kernels. Device
code uses shared-memory AES T-tables and a per-trial AES key schedule. The Python
CPU oracle uses cryptography/OpenSSL, whose hardware acceleration depends on the
installed backend and CPU. The scalar host AES derives batch keys only.

References:

- https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html
- https://docs.nvidia.com/cuda/archive/11.6.2/cuda-compiler-driver-nvcc/index.html

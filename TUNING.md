# Short GPU tuning and CPU AES-NI comparison

## Architecture: why more VRAM stopped helping

The original mapping is one thread per experiment.  At `m=4096`, `ell_i=16`
each experiment needs 6144 buckets, so the bins alone are 24 KiB of *private*
global memory per thread (plus 8 KiB for the PRP round cache in the cached
path).  Bucket lookups are uniformly random, so the 32 lanes of a warp touch 32
different 24 KiB regions: one warp instruction becomes up to 32 scattered
memory sectors for 128 bytes of useful data.  Throughput then scales with
resident threads, and resident threads cost tens of KiB each.  Two gigabytes of
scratch buys about 65k threads (roughly 600 threads per SM on a 108-SM GPU);
the next few gigabytes buy little because random-access latency, not capacity,
is already the limit.  That is the "tens of times the VRAM for little speedup"
effect.

The new `--table-layout shared` path changes the mapping instead of adding
memory:

* one warp owns one experiment; its 16-bit bucket table and 16-bit round cache
  live in shared memory (about 20 KiB per warp at `m=4096`, `ell_i=16`);
* all 32 lanes build the round cache in parallel, then lane 0 performs the
  inherently sequential insertion;
* random bucket and cache accesses stay on chip, so micro-launch size no
  longer allocates any per-trial global memory;
* a block hosts as many independent experiments as fit, typically 4 warps on a
  99 KiB consumer GPU and 7-8 on an A100/H100.

This does not change the experiment: same per-trial key, walk, candidate
choices, eviction rules and failure condition.  It requires `PNUM < 16384`,
`half_bits <= 12` and the round cache.  `auto` (default) uses it for
`PNUM >= 1024` when it fits and keeps the original per-thread mapping for small
tables; `--table-layout global` restores the old path.  The server log now
prints `table=shared-trial|cached-global|global`, `warps`, `shared_bytes`,
register/local usage and `active_blocks/SM`.

This is a bandwidth/latency argument, not a claim of a measured speedup.  Run
the comparison below on the target GPU before choosing a production setting.

## Two-million-trial CPU/GPU comparison

`compare_cpu_gpu.py` runs the same ET/BT/seed/`m` on the old direct GPU path,
the cached global path, the new warp-private shared path and native AES-NI CPU
(1 and 4 workers), then requires every configuration to produce exactly the
same failure set.  A small frozen-reference preflight runs first, so a broken
kernel is rejected before the long measurements.

```bash
cmake -S . -B build-cuda11 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda11 -j4
ctest --test-dir build-cuda11 --output-on-failure

python3 compare_cpu_gpu.py --out compare_E20_01
```

Defaults are `ET=20` (1,048,576 trials), `BT=20`, `m=4096`, `ell_i=16`,
`seed=123`, both modes, GPU micro 65536, and CPU workers 1 and 4.  Useful
variants:

```bash
# One mode only, three repeats of every GPU case
python3 compare_cpu_gpu.py --modes bgh25 --cpu-workers 1 4 --repeats 3 --out compare_bgh_01

# Only the new architecture and the CPU, skipping the legacy 256 MiB case
python3 compare_cpu_gpu.py --skip-legacy --modes bgh25 --out compare_shared_01

# Pin the number of independent experiments per block instead of auto
python3 compare_cpu_gpu.py --shared-warps 4 8 --skip-legacy --out compare_warps_01
```

The script writes `report.json` and `comparison.csv`, and prints trials/s plus
the actual table layout, warps and shared bytes for every case.  `work_seconds`
excludes GPU initialization and the one-second progress polling tail.  If the
shared layout is requested explicitly and does not fit, the run fails instead of
silently falling back.  For exhaustive 2^15 outcome checking of the shared
kernel, use:

```bash
python3 smoke_test.py --exe ./build-cuda11/cuckoo_bench --table-layout shared \
  --prp-cache on --out smoke_shared_01 --require-failures
```

The GPU table-space budget is configurable with `--scratch-mib` (per worker).
The default remains 256 MiB. Seeds, the v3 journal format and the existing
kernels are unchanged; the shared-trial kernel is additive and selected only by
`--table-layout shared|auto`. Changing workers, micro, scratch budget or table
layout is allowed when resuming the same experiment.

Build on the server with the existing CUDA 11.6 toolchain:

```bash
cmake -S . -B build-cuda11 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-cuda11 -j4
ctest --test-dir build-cuda11 --output-on-failure
```

To build only the native CPU executable, without recompiling the GPU executable:

```bash
cmake --build build-cuda11 --target cpu_bench -j4
```

## A few seconds of CPU measurement first

```bash
./build-cuda11/cpu_bench --mode bgh25 --pnum 4096 --ell-i 16 --et 12 --bt 12 --seed 123 --workers 1 --out cpu_E12_w1
./build-cuda11/cpu_bench --mode bgh25 --pnum 4096 --ell-i 16 --et 12 --bt 12 --seed 123 --workers 4 --out cpu_E12_w4
```

`cpu_bench` uses real x86 AES-NI, not Python. It uses the same PRP, trial keys,
walk choices, candidate buckets, packed table and insertion decisions as the GPU.
It checks CPU AES-NI availability and a NIST AES vector before running.
One coordinator owns the batch; helper CPU threads divide its trial offsets.
This allows multicore measurements even with a single batch. All helper results
must finish before the coordinator commits that batch. Pause discards an incomplete
batch; the same command resumes completed batches using the existing journal scheme.
CPU `--micro` is a scheduling grain capped at 256 trials between pause checks.

## Reproducible short comparison

```bash
python3 tune_bench.py --out tune_E16_01
```

Defaults: BGH25, m=4096, ell_i=16, seed=123, sequential, ET=BT=16 (65536 trials
per invocation). All configurations use exactly the same experiment IDs and keys:

| Case | Workers | Requested micro | Table budget per GPU worker |
|---|---:|---:|---:|
| GPU baseline | 1 | 16384 | 256 MiB |
| GPU larger micro | 1 | 32768 | 1024 MiB |
| GPU larger micro | 1 | 65536 | 2048 MiB |
| GPU extra worker (only when ET > BT) | 2 | 32768 | 1024 MiB |
| GPU shared-trial, auto warps | 1 | 65536 | 0 MiB (on-chip tables) |
| CPU AES-NI | 1 | CPU scheduling grain 64 | CPU memory only |
| CPU AES-NI | 4 | CPU scheduling grain 64 | CPU memory only |

For m=4096, BGH25 micro=32768/65536 needs 768/1536 MiB table memory.
The budget is not an allocation request: actual allocation also depends on micro,
batch size and bucket count. On a 16 GB GPU, do not request a full 16 GiB table;
CUDA context, buffers and other allocations need room.
With ET=BT=16 there is one batch, so the GPU extra-worker case is omitted.
For GPU worker scaling, use `--et 18 --bt 16` (four batches) after choosing the micro.

GPU cases repeat twice; CPU cases once. Change `--gpu-repeats` / `--cpu-repeats`
for more stable measurements. The script checks completion and exact failure-event
agreement, writes raw logs and journals, `report.json`, and `comparison.csv`, and
ranks the tested settings by median work throughput. The output directory must be new.
Different modes may have different best configurations. To include DRRT18:

```bash
python3 tune_bench.py --modes bgh25 drrt18 --out tune_both_E16_01
```

CPU-only short comparison (4096 trials per case):

```bash
python3 tune_bench.py --cpu-only --et 12 --bt 12 --out tune_cpu_E12_01
```

Optional `--cpu-workers 1 2 4 8` should reflect the CPU cores actually assigned
to the container. The report includes CPU affinity/quota and GPU identification
when available. Each invocation has a 600-second timeout, configurable with `--timeout`.

## Timing and validation

`elapsed_seconds` retains its old meaning: current invocation, including setup and
the progress polling wait. New `work_elapsed_seconds` measures worker launch through
the last worker's completion, including table allocation and journal commits, but
excluding GPU initialization/AES self-test and the one-second progress polling tail.
It is not a pure-kernel timer. Resumed runs include only new work time, so tuning
always uses fresh directories. The script records both timers and subprocess wall time;
subsecond work times should be confirmed with a larger ET.

CPU CTest compares every failure position in 32768 trials per mode with the independent
Python oracle (5 BGH25 failures, 214 DRRT18 failures), plus parallel affine cases and
pause behavior. The existing smoke test can also check the CPU backend:

```bash
python3 smoke_test.py --exe ./build-cuda11/cpu_bench --out smoke_cpu_01 --require-failures
```

The CPU backend passed these tests locally, including a real mid-run file pause and
resume.  The host CTest suite also emulates the warp-private shared layout on the
CPU (both modes and input modes, `m=16` and `m=4096`, non-zero micro offsets and
short tail blocks) and compares every table slot and failure decision with the
direct AES-NI reference.  The shared CUDA kernel itself still needs a server CUDA
build and run; no GPU speedup is assumed before `compare_cpu_gpu.py` measures it.
On this development machine, the CPU AES-NI cached path at `m=4096`, `ell_i=16`,
ET=20 needed 87.7 s for BGH25 with 4 workers (11,960 trials/s); that is the
number the GPU cases must beat on the server.

# Short GPU tuning and CPU AES-NI comparison

The GPU table-space budget is now configurable with `--scratch-mib` (per worker).
The default remains 256 MiB. Seeds, v3 journal format and GPU kernels are unchanged.
Changing workers, micro or scratch budget is allowed when resuming the same experiment.

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
resume. The GPU kernel body was preserved; the new GPU host budget/timing changes
still need an actual server CUDA build and run. No speedup is assumed before measurement.

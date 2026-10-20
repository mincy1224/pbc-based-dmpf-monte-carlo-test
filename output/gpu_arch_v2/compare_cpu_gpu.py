#!/usr/bin/env python3
"""Compare native AES-NI CPU with three GPU layouts on 2^20 identical trials.

GPU cases: the original per-thread global table, the cached global table and
the warp-private shared-memory table (one experiment per warp).  Every case
must reproduce exactly the same failure set, so timing differences cannot come
from changed experiment semantics.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

from tune_bench import read_events, require, write_json


def preflight(args, root):
    """Exercise successes/failures/tail blocks against frozen independent Python outcomes."""
    expected = json.loads(Path(__file__).with_name('smoke_expected.json').read_text())
    cases = []
    if not args.cpu_only:
        cases += [('gpu_direct', args.gpu_exe, 'cuda', 'off', 257, 128, 1, 'global', 0),
                  ('gpu_cached128', args.gpu_exe, 'cuda', 'on', 257, 128, 1, 'global', 0),
                  ('gpu_shared', args.gpu_exe, 'cuda', 'on', 257, 256, 1, 'shared', 0),
                  ('gpu_cached_tail', args.gpu_exe, 'cuda', 'on', 37, 256, 1, 'global', 0),
                  ('gpu_shared_tail', args.gpu_exe, 'cuda', 'on', 5, 256, 1, 'shared', 2)]
    if not args.gpu_only:
        cases += [('cpu_direct', args.cpu_exe, 'cpu-aesni', 'off', 64, 256, 1, 'global', 0),
                  ('cpu_cached', args.cpu_exe, 'cpu-aesni', 'on', 64, 256, 4, 'global', 0)]
    results = []
    for mode in dict.fromkeys(args.modes):
        for name, exe, backend, cache, micro, block, workers, layout, warps in cases:
            directory = root / f'preflight_{mode}_{name}'
            command = [str(exe), '--mode', mode, '--et', '13', '--bt', '8', '--pnum', '16',
                '--ell-i', '8', '--seed', '123', '--input-mode', 'sequential',
                '--workers', str(workers), '--micro', str(micro), '--scratch-mib', '64',
                '--prp-cache', cache, '--block-threads', str(block),
                '--table-layout', layout, '--out', str(directory)]
            if warps:
                command += ['--warps-per-block', str(warps)]
            print(f'PREFLIGHT {mode} {name}: 8192 fixed reference trials', flush=True)
            with directory.with_suffix('.log').open('w', encoding='utf-8') as log:
                log.write(json.dumps(command) + '\n'); log.flush()
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
            summary = json.loads((directory / 'summary.json').read_text())
            for key, value in dict(status='complete', completed=8192, planned=8192, ET=13, BT=8,
                    mode=mode, PNUM_T=16, ELL_I=8, seed_decimal='123', input_mode='sequential',
                    seed_scheme='aes-batch-v1', backend=backend, prp_cache_enabled=cache == 'on',
                    table_layout_requested=layout, completed_batches=32, pending_batches=0).items():
                require(summary.get(key) == value, f'Preflight {mode} {name}: invalid {key}')
            events = read_events(directory, 8192, 8)
            require(events == set(expected['failures'][mode]) and summary['failed'] == len(events),
                    f'Preflight failed for {mode} {name}; do not use these measurements')
            results.append(dict(mode=mode, case=name, completed=8192, failed=len(events)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-exe', type=Path, default=Path('build-cuda11/cuckoo_bench'))
    parser.add_argument('--cpu-exe', type=Path, default=Path('build-cuda11/cpu_bench'))
    parser.add_argument('--out', type=Path, default=Path('compare_E20'))
    parser.add_argument('--modes', nargs='+', choices=['bgh25', 'drrt18'], default=['bgh25', 'drrt18'])
    parser.add_argument('--et', type=int, default=20)
    parser.add_argument('--bt', type=int, help='Default min(ET,20); fixed across every configuration')
    parser.add_argument('--pnum', type=int, default=4096)
    parser.add_argument('--ell-i', type=int, default=16)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--input-mode', choices=['sequential', 'affine'], default='sequential')
    parser.add_argument('--cpu-workers', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--gpu-blocks', nargs='+', type=int, choices=[64, 128, 256], default=[128, 256])
    parser.add_argument('--shared-warps', nargs='+', type=int, default=[0],
                        help='Trials per block in the warp-private shared layout; 0=auto')
    parser.add_argument('--micro', type=int, default=65536)
    parser.add_argument('--scratch-mib', type=int, default=2048)
    parser.add_argument('--skip-shared', action='store_true', help='Skip the warp-private shared-memory case')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=7200, help='Seconds per complete invocation')
    parser.add_argument('--include-large-legacy', action='store_true', help='Also test old GPU with about 14 GiB of tables')
    parser.add_argument('--include-direct-cpu', action='store_true', help='Also time the CPU without the exact round cache')
    parser.add_argument('--skip-legacy', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--gpu-only', action='store_true')
    parser.add_argument('--skip-preflight', action='store_true', help='Only after a successful preflight of the same executables')
    args = parser.parse_args()
    args.bt = min(args.et, 20) if args.bt is None else args.bt
    require(not (args.cpu_only and args.gpu_only), 'Choose only one backend-only option')
    require(1 <= args.bt <= min(args.et, 30) and args.et <= 40, 'Invalid ET/BT')
    require(3 <= args.pnum <= 65534 and 0 <= args.ell_i <= 61 and (1 << args.ell_i) >= args.pnum,
            'Invalid pnum/ell-i')
    require(0 <= args.seed < (1 << 64) and args.repeats > 0 and args.timeout > 0, 'Invalid seed/repeats/timeout')
    require(0 < args.micro < (1 << 32) and 0 < args.scratch_mib < (1 << 32), 'Invalid GPU memory arguments')
    require(all(1 <= n <= 256 for n in args.cpu_workers), 'Invalid CPU worker count')
    require(all(0 <= n <= 32 for n in args.shared_warps), 'Invalid shared warp count')
    require(not args.include_large_legacy or args.bt >= 20, '14 GiB case needs BT >=20')
    args.cpu_exe, args.gpu_exe = args.cpu_exe.resolve(), args.gpu_exe.resolve()
    if not args.cpu_only:
        require(args.gpu_exe.is_file(), f'GPU executable missing: {args.gpu_exe}')
    if not args.gpu_only:
        require(args.cpu_exe.is_file(), f'CPU executable missing: {args.cpu_exe}')
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    total = 1 << args.et
    # name, backend, workers, requested micro, scratch MiB, round cache, block size,
    # table layout, warp-private trials per block (0 = auto)
    cases = []
    if not args.cpu_only:
        if not args.skip_legacy:
            cases.append(('gpu_direct_256m', 'cuda', 1, 16384, 256, 'off', 128, 'global', 0))
        if args.include_large_legacy:
            cases.append(('gpu_direct_14g', 'cuda', 1, 611669, 16384, 'off', 128, 'global', 0))
        for block in dict.fromkeys(args.gpu_blocks):
            cases.append((f'gpu_cached_b{block}', 'cuda', 1, args.micro, args.scratch_mib,
                          'on', block, 'global', 0))
        if not args.skip_shared:
            for warps in dict.fromkeys(args.shared_warps):
                name = 'gpu_shared_auto' if warps == 0 else f'gpu_shared_w{warps}'
                cases.append((name, 'cuda', 1, args.micro, 64, 'on', 256, 'shared', warps))
    if not args.gpu_only:
        for workers in dict.fromkeys(args.cpu_workers):
            cases.append((f'cpu_cached_w{workers}', 'cpu-aesni', workers, 64, 256, 'on', 256,
                          'global', 0))
        if args.include_direct_cpu:
            cases.append(('cpu_direct_w1', 'cpu-aesni', 1, 64, 256, 'off', 256, 'global', 0))
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = dict(status='running', settings=settings, platform=platform.platform(),
                  logical_cpus=os.cpu_count(), records=[], comparisons={})
    if hasattr(os, 'sched_getaffinity'):
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    for filename in ('/proc/cpuinfo', '/sys/fs/cgroup/cpu.max',
                     '/sys/fs/cgroup/cpu/cpu.cfs_quota_us', '/sys/fs/cgroup/cpu/cpu.cfs_period_us'):
        try:
            data = Path(filename).read_text()
            if filename == '/proc/cpuinfo':
                report['cpu_model'] = next((line.split(':', 1)[1].strip() for line in data.splitlines()
                                            if line.startswith('model name')), '')
            else:
                report[filename] = data.strip()
        except OSError:
            pass
    try:
        report['gpu_info'] = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
            '--format=csv,noheader'], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    report_path = root / 'report.json'
    write_json(report_path, report)
    print(f'Each invocation: {total} trials, m={args.pnum}, ET={args.et}, BT={args.bt}. Runs are sequential.', flush=True)
    try:
        if not args.skip_preflight:
            report['preflight'] = preflight(args, root)
            write_json(report_path, report)
        for mode in args.modes:
            reference = None
            reference_name = None
            by_case = {}
            for name, backend, workers, micro, scratch, cache, block, layout, warps in cases:
                rates = []
                for repeat in range(1, args.repeats + 1):
                    directory = root / f'{mode}_{name}_r{repeat}'
                    exe = args.gpu_exe if backend == 'cuda' else args.cpu_exe
                    command = [str(exe), '--mode', mode, '--et', str(args.et), '--bt', str(args.bt),
                        '--pnum', str(args.pnum), '--ell-i', str(args.ell_i), '--seed', str(args.seed),
                        '--input-mode', args.input_mode, '--workers', str(workers), '--micro', str(micro),
                        '--scratch-mib', str(scratch), '--prp-cache', cache, '--block-threads', str(block),
                        '--table-layout', layout, '--stop-after', str(total), '--out', str(directory)]
                    if warps:
                        command += ['--warps-per-block', str(warps)]
                    logfile = directory.with_suffix('.log')
                    print(f'RUN {mode} {name} {repeat}/{args.repeats} (details: {logfile.name})', flush=True)
                    started = time.monotonic()
                    with logfile.open('w', encoding='utf-8') as log:
                        log.write(json.dumps(command) + '\n'); log.flush()
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
                    wall = time.monotonic() - started
                    summary = json.loads((directory / 'summary.json').read_text())
                    expected = dict(status='complete', completed=total, planned=total, backend=backend,
                        mode=mode, ET=args.et, BT=args.bt, PNUM_T=args.pnum, ELL_I=args.ell_i,
                        seed_decimal=str(args.seed), input_mode=args.input_mode, workers=workers,
                        prp_cache_enabled=cache == 'on', seed_scheme='aes-batch-v1',
                        table_layout_requested=layout,
                        completed_batches=1 << (args.et - args.bt), pending_batches=0)
                    for key, value in expected.items():
                        require(summary.get(key) == value, f'{name}: unexpected {key}: {summary.get(key)!r}')
                    work = summary.get('work_elapsed_seconds', 0)
                    require(work > 0, 'Missing work timer; use a fresh output directory and rebuilt executables')
                    events = read_events(directory, total, args.bt)
                    require(len(events) == summary['failed'], f'{name}: failure summary disagrees with export')
                    if reference is None:
                        reference, reference_name = events, name
                    mismatch = events.symmetric_difference(reference)
                    with directory.with_suffix('.mismatches.csv').open('w', newline='') as stream:
                        writer = csv.writer(stream); writer.writerow(['trial_id', 'reference_failed', 'case_failed'])
                        for trial in sorted(mismatch):
                            writer.writerow([trial, int(trial in reference), int(trial in events)])
                    text = logfile.read_text(errors='replace')
                    actual_micros = [int(n) for n in re.findall(r'Worker .*? micro=(\d+)', text)]
                    actual_layouts = re.findall(r'table=(\S+)', text)
                    actual_warps = [int(n) for n in re.findall(r'\bwarps=(\d+)', text)]
                    actual_shared = [int(n) for n in re.findall(r'shared_bytes=(\d+)', text)]
                    actual_scratch = [int(n) for n in re.findall(r'scratch_bytes=(\d+)', text)]
                    if layout == 'shared':
                        require(actual_layouts and actual_layouts[0] == 'shared-trial',
                                f'{name}: shared layout requested but not used ({actual_layouts[:1]})')
                    row = dict(mode=mode, case=name, backend=backend, workers=workers, cache=cache, block=block,
                        table_layout=layout, warps_requested=warps,
                        table_layout_actual=actual_layouts[0] if actual_layouts else None,
                        warps_actual=actual_warps[0] if actual_warps else None,
                        shared_bytes_actual=actual_shared[0] if actual_shared else None,
                        scratch_bytes_actual=actual_scratch[0] if actual_scratch else None,
                        micro_requested=micro, actual_gpu_micro=actual_micros[0] if actual_micros else None,
                        scratch_mib=scratch, repeat=repeat, completed=total, failed=len(events), mismatches=len(mismatch),
                        work_seconds=work, work_trials_per_second=total / work,
                        elapsed_seconds=summary['elapsed_seconds'], process_wall_seconds=wall)
                    report['records'].append(row); write_json(report_path, report)
                    require(not mismatch, f'{name}: {len(mismatch)} outcomes differ from {reference_name}; see mismatch CSV')
                    rates.append(total / work)
                    print(f'  work={work:.3f}s, {total / work:.1f} trials/s, failed={len(events)}, mismatches=0', flush=True)
                by_case[name] = dict(backend=backend, workers=workers,
                    median_trials_per_second=statistics.median(rates),
                    table_layout=row['table_layout_actual'], warps=row['warps_actual'],
                    shared_bytes=row['shared_bytes_actual'],
                    scratch_bytes=row['scratch_bytes_actual'])
            gpu = {name: row['median_trials_per_second'] for name, row in by_case.items() if row['backend'] == 'cuda'}
            cpu = {name: row['median_trials_per_second'] for name, row in by_case.items() if row['backend'] == 'cpu-aesni'}
            comparison = dict(cases=by_case)
            if gpu and cpu:
                best_gpu, best_cpu = max(gpu, key=gpu.get), max(cpu, key=cpu.get)
                comparison.update(best_gpu=best_gpu, best_cpu=best_cpu,
                    gpu_over_best_cpu=gpu[best_gpu] / cpu[best_cpu],
                    gpu_over_each_cpu={name: gpu[best_gpu] / rate for name, rate in cpu.items()})
                print(f'{mode}: best tested GPU / best tested CPU = {comparison["gpu_over_best_cpu"]:.3f}x', flush=True)
            if 'gpu_direct_256m' in gpu:
                comparison['cached_gpu_over_direct'] = {name: rate / gpu['gpu_direct_256m']
                    for name, rate in gpu.items() if name.startswith('gpu_cached')}
                comparison['shared_gpu_over_direct'] = {name: rate / gpu['gpu_direct_256m']
                    for name, rate in gpu.items() if name.startswith('gpu_shared')}
            report['comparisons'][mode] = comparison
            write_json(report_path, report)
            for name, row in by_case.items():
                print(f"  {name:22s} {row['median_trials_per_second']:12.1f} trials/s"
                      f"  layout={row['table_layout'] or '-'} warps={row['warps'] or '-'}"
                      f" shared={row['shared_bytes'] or 0}B"
                      f" scratch={row['scratch_bytes'] or 0}B", flush=True)
        with (root / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report['records'][0])); writer.writeheader()
            writer.writerows(report['records'])
        report['status'] = 'passed'; write_json(report_path, report)
        print(f'PASSED: all {total} per-mode outcomes agree across all configurations. {root}', flush=True)
        return 0
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        write_json(report_path, report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())

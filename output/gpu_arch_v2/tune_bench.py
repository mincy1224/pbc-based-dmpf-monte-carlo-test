#!/usr/bin/env python3
"""Short, reproducible GPU tuning and native CPU AES-NI comparison (no Python AES required)."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def read_events(directory, total, bt):
    events = set()
    with (directory / 'failures.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['batch_id', 'offset'], 'Bad failure export header')
        for row in reader:
            batch, offset = int(row['batch_id']), int(row['offset'])
            require(0 <= batch < (total >> bt) and 0 <= offset < (1 << bt), 'Out-of-range failure')
            event = batch * (1 << bt) + offset
            require(event not in events, 'Duplicate failure')
            events.add(event)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-exe', type=Path, default=Path('build-cuda11/cuckoo_bench'))
    parser.add_argument('--cpu-exe', type=Path, default=Path('build-cuda11/cpu_bench'))
    parser.add_argument('--out', type=Path, default=Path('tune_E16'))
    parser.add_argument('--et', type=int, default=16)
    parser.add_argument('--bt', type=int, default=16)
    parser.add_argument('--pnum', type=int, default=4096)
    parser.add_argument('--ell-i', type=int, default=16)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--input-mode', choices=['sequential', 'affine'], default='sequential')
    parser.add_argument('--modes', nargs='+', choices=['bgh25', 'drrt18'], default=['bgh25'])
    parser.add_argument('--cpu-workers', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--gpu-repeats', type=int, default=2)
    parser.add_argument('--cpu-repeats', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=600, help='Seconds per invocation')
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--gpu-only', action='store_true')
    args = parser.parse_args()
    require(not (args.cpu_only and args.gpu_only), 'Choose only one backend-only option')
    require(1 <= args.bt <= min(args.et, 30) and args.et <= 40, 'Invalid ET/BT')
    require(args.cpu_only or args.bt >= 16, 'GPU tuning needs BT >= 16 to test micro=65536')
    require(args.cpu_repeats > 0 and args.gpu_repeats > 0 and args.timeout > 0, 'Invalid repeats/timeout')
    require(all(1 <= n <= 256 for n in args.cpu_workers), 'Invalid CPU worker count')
    args.gpu_exe, args.cpu_exe = args.gpu_exe.resolve(), args.cpu_exe.resolve()
    for required in ([args.gpu_exe] if args.gpu_only else [args.cpu_exe] if args.cpu_only else [args.gpu_exe, args.cpu_exe]):
        require(required.is_file(), f'Executable missing: {required}')
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    total = 1 << args.et
    cases = []
    if not args.cpu_only:
        cases += [('gpu_256m', 'cuda', 1, 16384, 256, 'global'),
                  ('gpu_micro32768', 'cuda', 1, 32768, 1024, 'global'),
                  ('gpu_micro65536', 'cuda', 1, 65536, 2048, 'global'),
                  ('gpu_shared', 'cuda', 1, 65536, 64, 'shared')]
        if args.et > args.bt:
            cases.append(('gpu_workers2', 'cuda', 2, 32768, 1024, 'global'))
    if not args.gpu_only:
        cases += [(f'cpu_w{n}', 'cpu-aesni', n, 64, 256, 'global')
                  for n in dict.fromkeys(args.cpu_workers)]
    report = dict(status='running', settings=vars(args).copy(), platform=platform.platform(),
                  logical_cpus=os.cpu_count(), records=[], aggregates=[])
    report['settings'] = {key: str(value) if isinstance(value, Path) else value for key, value in report['settings'].items()}
    if hasattr(os, 'sched_getaffinity'):
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    for location in ('/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
                     '/sys/fs/cgroup/cpu/cpu.cfs_period_us'):
        path = Path(location)
        if path.is_file():
            report[location] = path.read_text().strip()
    if Path('/proc/cpuinfo').is_file():
        report['cpu_model'] = next((line.split(':', 1)[1].strip()
                                   for line in Path('/proc/cpuinfo').read_text().splitlines()
                                   if line.startswith('model name')), '')
    try:
        report['nvidia_smi'] = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    report_path = root / 'report.json'
    write_json(report_path, report)
    try:
        for mode in args.modes:
            reference = None
            for name, backend, workers, micro, scratch, layout in cases:
                rates, durations = [], []
                repeats = args.gpu_repeats if backend == 'cuda' else args.cpu_repeats
                for repeat in range(1, repeats + 1):
                    directory = root / f'{mode}_{name}_r{repeat}'
                    exe = args.gpu_exe if backend == 'cuda' else args.cpu_exe
                    cache = 'off' if name == 'gpu_256m' else 'on'
                    command = [str(exe), '--mode', mode, '--et', str(args.et), '--bt', str(args.bt),
                               '--pnum', str(args.pnum), '--ell-i', str(args.ell_i),
                               '--seed', str(args.seed), '--input-mode', args.input_mode,
                               '--workers', str(workers), '--micro', str(micro),
                               '--scratch-mib', str(scratch), '--prp-cache', cache,
                               '--table-layout', layout,
                               '--stop-after', str(total), '--out', str(directory)]
                    print(f'RUN {mode} {name} repeat={repeat}/{repeats}, trials={total}', flush=True)
                    started = time.monotonic()
                    with directory.with_suffix('.log').open('w', encoding='utf-8') as log:
                        log.write(json.dumps(command) + '\n'); log.flush()
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       check=True, timeout=args.timeout)
                    wall = time.monotonic() - started
                    summary = json.loads((directory / 'summary.json').read_text())
                    expected = dict(status='complete', completed=total, planned=total, backend=backend,
                                    ET=args.et, BT=args.bt, PNUM_T=args.pnum, ELL_I=args.ell_i,
                                    seed_decimal=str(args.seed), mode=mode, input_mode=args.input_mode,
                                    workers=workers, seed_scheme='aes-batch-v1', prp_cache_enabled=cache == 'on',
                                    table_layout_requested=layout,
                                    completed_batches=1 << (args.et - args.bt), pending_batches=0)
                    for key, value in expected.items():
                        require(summary.get(key) == value, f'{name}: unexpected {key}: {summary.get(key)!r}')
                    elapsed = summary.get('work_elapsed_seconds', 0)
                    require(elapsed > 0, 'Missing work timer: update and rebuild both executables')
                    events = read_events(directory, total, args.bt)
                    require(len(events) == summary['failed'], 'Failure count disagrees with CSV')
                    if reference is None:
                        reference = events
                    require(events == reference, f'{name}: failure positions differ from other configurations')
                    rates.append(total / elapsed); durations.append(elapsed)
                    row = dict(mode=mode, case=name, repeat=repeat, backend=backend, workers=workers,
                               table_layout=layout,
                               micro=micro, scratch_mib=scratch, trials=total, failed=len(events),
                               work_seconds=elapsed, work_trials_per_second=total / elapsed,
                               elapsed_seconds=summary['elapsed_seconds'], process_wall_seconds=wall,
                               subsecond_timing=elapsed < 1.0)
                    report['records'].append(row)
                    write_json(report_path, report)
                    print(f'  {elapsed:.3f}s work; {total / elapsed:.1f} trials/s; failed={len(events)}', flush=True)
                report['aggregates'].append(dict(mode=mode, case=name, backend=backend, workers=workers,
                    median_work_seconds=statistics.median(durations),
                    median_trials_per_second=statistics.median(rates)))
            ranked = sorted((row for row in report['aggregates'] if row['mode'] == mode),
                            key=lambda row: row['median_trials_per_second'], reverse=True)
            print(f'\n{mode}: best tested configuration first', flush=True)
            for row in ranked:
                print(f"  {row['case']:20s} {row['median_trials_per_second']:12.1f} trials/s", flush=True)
        with (root / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report['records'][0]))
            writer.writeheader(); writer.writerows(report['records'])
        report['status'] = 'passed'
        write_json(report_path, report)
        print(f'PASSED: all failure positions agree. Results: {root}', flush=True)
        return 0
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        write_json(report_path, report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exhaustive CPU/GPU outcome comparison for 2^15 trials per cuckoo mode.

Requires Python 3.9+, cryptography, and the compiled CUDA executable.
The CPU oracle runs independently; GPU success is inferred only after verifying
that all trials were completed and the sparse failure export is well formed.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

from verify_cpu import Oracle, batch_encryptor, params

ET, BT = 15, 8
TOTAL, BATCH_SIZE = 1 << ET, 1 << BT
MODES = ('bgh25', 'drrt18')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def read_results(directory, args, mode, complete=True):
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    buckets, width, domain, _, kicks = params(mode, args.pnum, args.ell_i)
    expected = dict(mode=mode, input_mode=args.input_mode, seed_decimal=str(args.seed),
                    ET=ET, BT=BT, PNUM_T=args.pnum, ELL_I=args.ell_i,
                    planned=TOTAL, BUCKETS_NUM=buckets, BUCKET_SIZE=width,
                    PRP_DOMAIN=domain, MAX_KICK_TIMES=kicks, seed_scheme='aes-batch-v1')
    for key, value in expected.items():
        require(summary.get(key) == value, f'{directory}: unexpected {key}: {summary.get(key)!r}')
    completed = summary['completed']
    require(0 <= completed <= TOTAL and completed % BATCH_SIZE == 0,
            f'{directory}: partial or invalid completed count')
    require(summary['completed_batches'] == completed // BATCH_SIZE and
            summary['pending_batches'] == (TOTAL - completed) // BATCH_SIZE,
            f'{directory}: inconsistent batch counts')
    if complete:
        require(summary['status'] == 'complete' and completed == TOTAL,
                f'{directory}: run did not complete all {TOTAL} trials')
    else:
        require(summary['status'] == 'paused' and 0 < completed < TOTAL,
                f'{directory}: pause did not leave both committed and pending batches')
    failed = set()
    with (directory / 'failures.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['batch_id', 'offset'], f'{directory}: invalid CSV header')
        for row in reader:
            batch, offset = int(row['batch_id']), int(row['offset'])
            require(0 <= batch < TOTAL // BATCH_SIZE and 0 <= offset < BATCH_SIZE,
                    f'{directory}: failure outside experiment range: {row}')
            trial = batch * BATCH_SIZE + offset
            require(trial not in failed, f'{directory}: duplicate failure {row}')
            failed.add(trial)
    require(len(failed) == summary['failed'] <= completed,
            f'{directory}: summary/CSV failure counts disagree')
    return summary, failed


def command(args, mode, directory, workers, micro, global_table=False):
    result = [str(args.exe), '--mode', mode, '--et', str(ET), '--bt', str(BT),
              '--pnum', str(args.pnum), '--ell-i', str(args.ell_i),
              '--seed', str(args.seed), '--input-mode', args.input_mode,
              '--workers', str(workers), '--micro', str(micro),
              '--stop-after', str(TOTAL), '--out', str(directory)]
    if global_table:
        result.append('--global-table')
    return result


def run(args, cmd, logfile):
    print(f'GPU: {logfile.stem}', flush=True)
    with logfile.open('w', encoding='utf-8') as log:
        log.write(json.dumps(cmd) + '\n')
        log.flush()
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=args.timeout)


def pause_run(args, mode, directory):
    # Many small launches make a real mid-run pause observable even on a fast GPU.
    # Wait for a committed record, then pause; resumed work changes both workers/micro.
    cmd = command(args, mode, directory, 1, 1)
    logfile = directory.parent / 'pause.log'
    print('GPU: pause after first committed batch, then resume', flush=True)
    with logfile.open('w', encoding='utf-8') as log:
        log.write(json.dumps(cmd) + '\n')
        log.flush()
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + args.timeout
            while process.poll() is None:
                # v3 file header is 56 bytes; a complete empty-failure record is 24.
                if any(path.stat().st_size >= 80 for path in directory.glob('worker_*.bin')):
                    (directory / 'PAUSE').touch()
                    break
                require(time.monotonic() < deadline, 'Timed out waiting for first batch')
                time.sleep(0.005)
            process.wait(timeout=max(1, deadline - time.monotonic()))
            require(process.returncode == 0, f'Pause run failed; see {logfile}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    paused, _ = read_results(directory, args, mode, complete=False)
    write_json(directory.parent / 'paused_summary.json', paused)
    (directory / 'PAUSE').unlink()
    run(args, command(args, mode, directory, 2, 113), directory.parent / 'resume.log')
    return paused


def cpu_results(args, mode, directory):
    failed = set()
    started = time.monotonic()
    path = directory / 'cpu_trials.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['trial_id', 'batch_id', 'offset', 'cpu_failed'])
        for batch in range(TOTAL // BATCH_SIZE):
            aes = batch_encryptor(args.seed, batch)
            for offset in range(BATCH_SIZE):
                trial = batch * BATCH_SIZE + offset
                result = Oracle(mode, args.pnum, args.ell_i, aes, offset,
                                args.input_mode).trial_failed()
                if result:
                    failed.add(trial)
                writer.writerow([trial, batch, offset, int(result)])
            stream.flush()
            if (batch + 1) % 4 == 0:
                print(f'CPU {mode}: {(batch + 1) * BATCH_SIZE}/{TOTAL}, '
                      f'failed={len(failed)}, elapsed={time.monotonic() - started:.1f}s', flush=True)
    return failed, time.monotonic() - started


def compare(directory, cpu, gpu_sets):
    mismatch_count = 0
    fields = ['trial_id', 'batch_id', 'offset', 'cpu_failed',
              *[name + '_failed' for name in gpu_sets], 'match']
    with (directory / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream, \
            (directory / 'mismatches.csv').open('w', newline='', encoding='utf-8') as errors:
        writer, bad = csv.writer(stream), csv.writer(errors)
        writer.writerow(fields)
        bad.writerow(fields)
        for trial in range(TOTAL):
            truth = trial in cpu
            values = [trial in failed for failed in gpu_sets.values()]
            match = all(value == truth for value in values)
            row = [trial, trial // BATCH_SIZE, trial % BATCH_SIZE,
                   int(truth), *map(int, values), int(match)]
            writer.writerow(row)
            if not match:
                bad.writerow(row)
                mismatch_count += 1
    return mismatch_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, default=Path('build/cuckoo_bench'))
    parser.add_argument('--out', type=Path, default=Path('smoke_2p15'))
    parser.add_argument('--pnum', type=int, default=16)
    parser.add_argument('--ell-i', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--input-mode', choices=['sequential', 'affine'], default='sequential')
    parser.add_argument('--timeout', type=float, default=600,
                        help='Timeout per GPU invocation in seconds (CPU oracle has no timeout)')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Generate CPU reference results only; does not certify a GPU smoke test')
    parser.add_argument('--require-failures', action='store_true',
                        help='Also require observed failures in each mode (useful for small cases)')
    args = parser.parse_args()
    require(3 <= args.pnum <= 65534 and 0 <= args.ell_i <= 61 and
            (1 << args.ell_i) >= args.pnum, 'Invalid pnum/ell-i')
    require(0 <= args.seed < (1 << 64) and args.timeout > 0, 'Invalid seed/timeout')
    args.exe = args.exe.resolve()
    if not args.cpu_only:
        require(args.exe.is_file(), f'CUDA executable not found: {args.exe}')
    args.out = args.out.resolve()
    # Refuse reuse: previous smoke results must never make a fresh run pass.
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', trials_per_mode=TOTAL, ET=ET, BT=BT,
                  pnum=args.pnum, ell_i=args.ell_i, seed=str(args.seed),
                  input_mode=args.input_mode, cpu_only=args.cpu_only,
                  require_failures=args.require_failures, modes={})
    write_json(args.out / 'report.json', report)
    try:
        failures = []
        for mode in MODES:
            directory = args.out / mode
            directory.mkdir()
            cpu, seconds = cpu_results(args, mode, directory)
            entry = dict(cpu_failed=len(cpu), cpu_seconds=seconds)
            report['modes'][mode] = entry
            write_json(args.out / 'report.json', report)
            if args.cpu_only:
                continue
            gpu_sets = {}
            run(args, command(args, mode, directory / 'baseline', 1, 257), directory / 'baseline.log')
            baseline, gpu_sets['gpu_baseline'] = read_results(directory / 'baseline', args, mode)
            run(args, command(args, mode, directory / 'global_multi', 2, 37, True), directory / 'global_multi.log')
            global_summary, gpu_sets['gpu_global_multi'] = read_results(directory / 'global_multi', args, mode)
            paused = pause_run(args, mode, directory / 'resumed')
            resumed, gpu_sets['gpu_resumed'] = read_results(directory / 'resumed', args, mode)
            # Rerunning a complete directory must perform no extra trials.
            run(args, command(args, mode, directory / 'resumed', 1, 19), directory / 'rerun_complete.log')
            _, again = read_results(directory / 'resumed', args, mode)
            require(again == gpu_sets['gpu_resumed'], 'Completed-run replay changed results')
            mismatches = compare(directory, cpu, gpu_sets)
            entry.update(mismatches=mismatches, baseline=baseline, global_multi=global_summary,
                         paused=paused, resumed=resumed,
                         failure_path_covered=bool(cpu))
            write_json(args.out / 'report.json', report)
            if mismatches:
                failures.append(f'{mode}: {mismatches} mismatches; see comparison.csv/mismatches.csv')
            if args.require_failures and not cpu:
                failures.append(f'{mode}: no failures observed; failure-path coverage is insufficient')
            if not mismatches and (cpu or not args.require_failures):
                print(f'PASS {mode}: all {TOTAL} outcomes match CPU in all 3 GPU runs', flush=True)
        require(not failures, '; '.join(failures))
        report['status'] = 'cpu_reference_only' if args.cpu_only else 'passed'
        write_json(args.out / 'report.json', report)
        print(f"{report['status'].upper()}: {args.out}", flush=True)
        return 0
    except BaseException as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        write_json(args.out / 'report.json', report)
        raise


if __name__ == '__main__':
    sys.exit(main())

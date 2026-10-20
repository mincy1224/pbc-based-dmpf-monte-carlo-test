#!/usr/bin/env python3
"""Slow independent CPU oracle for the CUDA prp_cuckoo experiment.

Only for tiny test runs, not for throughput measurement. Needs cryptography.
"""
import argparse
import csv
import math
import struct
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

MASK64 = (1 << 64) - 1


def aes_cipher(key):
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return enc.update


def words(*v):
    return b''.join(x.to_bytes(4, 'big') for x in v)


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def original_log2_u64(n):
    e = n.bit_length() - 1
    x = n / (1 << e)
    if x >= 2:
        return f32(e + 1)
    z = (x - 1) / (x + 1)
    z2 = z * z
    s, term = 0.0, z
    for k in range(1, 24, 2):
        s += term / k
        term *= z2
    return f32(e + 2.0 * s * 1.4426950408889634074)


def params(mode, pnum, ell_i):
    if mode == 'bgh25':
        buckets = pnum * (200 if pnum < 30 else 150) // 100
        maxkick = 3 * pnum
    else:
        val = f32(170.0 + original_log2_u64(pnum))
        buckets = math.ceil(f32(pnum * (val / 123.5)))
        maxkick = 100
    bucket_size = ((1 << ell_i) * 3 + buckets - 1) // buckets
    domain = buckets * bucket_size
    bits = max(2, (domain - 1).bit_length())
    if bits & 1:
        bits += 1
    half = bits // 2
    return buckets, bucket_size, domain, half, maxkick


def splitmix_next(state):
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return state, z ^ (z >> 31)


def batch_encryptor(seed, batch):
    master = seed.to_bytes(8, 'little') + (seed ^ 0x9E3779B97F4A7C15).to_bytes(8, 'little')
    data = aes_cipher(master)(words(0x42415443, batch >> 32, batch & 0xffffffff, 0x50524731))
    lo = seed ^ int.from_bytes(data[:8], 'big')
    hi = int.from_bytes(data[8:], 'big') ^ (seed ^ 0x243F6A8885A308D3)
    return aes_cipher(lo.to_bytes(8, 'little') + hi.to_bytes(8, 'little'))


class Oracle:
    def __init__(self, mode, pnum, ell_i, batch_aes, trial, input_mode):
        self.buckets, self.width, self.domain, self.half, self.maxkick = params(mode, pnum, ell_i)
        self.pnum = pnum
        self.mode = mode
        self.mask = (1 << self.half) - 1
        self.range_mask = (1 << ell_i) - 1
        self.prp_aes = aes_cipher(batch_aes(words(0x4b455931, 0, trial, 0x50525031)))
        w = batch_aes(words(0x57414c4b, 0, trial, 0x524e4731))
        self.walk_state = int.from_bytes(w[:8], 'big')
        self.shift = int.from_bytes(w[8:], 'big') & self.range_mask
        state, val = splitmix_next(self.shift ^ 0xd1b54a32d192ed03)
        self.step = (val | 1) & self.range_mask
        self.walk_state, self.bucket_randomness = splitmix_next(self.walk_state)
        self.input_mode = input_mode
        self.table = [None] * self.buckets

    def element(self, ordinal):
        if self.input_mode == 'sequential':
            return ordinal
        return (self.shift + ordinal * self.step) & self.range_mask

    def bucket(self, ordinal, which):
        y = 3 * self.element(ordinal) + which
        while True:
            left, right = y >> self.half, y & self.mask
            for r in range(8):
                inp = words(0x43455350, r, self.half, right & 0xffffffff)
                ciphertext = self.prp_aes(inp)
                left, right = right, (left ^ (int.from_bytes(ciphertext[-4:], 'big') & self.mask)) & self.mask
            y = (left << self.half) | right
            if y < self.domain:
                return y // self.width

    def choose_bgh(self):
        if self.bucket_randomness < 3:
            self.walk_state, self.bucket_randomness = splitmix_next(self.walk_state)
        which = self.bucket_randomness % 3
        self.bucket_randomness //= 3
        return which

    def trial_failed(self):
        for item in range(self.pnum):
            if self.mode == 'bgh25':
                cur = item
                collisions = 0
                while True:
                    which = self.choose_bgh()
                    bucket = self.bucket(cur, which)
                    displaced = self.table[bucket]
                    if displaced is None:
                        self.table[bucket] = (cur, which)
                        break
                    collisions += 1
                    if collisions > self.maxkick:
                        return True
                    self.table[bucket] = (cur, which)
                    cur = displaced[0]
            else:
                initial = [self.bucket(item, k) for k in range(3)]
                empty_k = next((k for k, bucket in enumerate(initial)
                                if self.table[bucket] is None), None)
                if empty_k is not None:
                    self.table[initial[empty_k]] = (item, empty_k)
                    continue
                cur, choice, bucket, kicks = item, 0, initial[0], 0
                while True:
                    displaced = self.table[bucket]
                    if displaced is None:
                        self.table[bucket] = (cur, choice)
                        break
                    if kicks >= self.maxkick:
                        return True
                    self.table[bucket] = (cur, choice)
                    cur, choice = displaced
                    previous_bucket = bucket
                    found = False
                    for _ in range(3):
                        choice = (choice + 1) % 3
                        target = self.bucket(cur, choice)
                        if target != previous_bucket:
                            bucket, found = target, True
                            break
                    if not found:
                        return True
                    kicks += 1
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['bgh25', 'drrt18'])
    ap.add_argument('--ell-i', type=int, default=8)
    ap.add_argument('--pnum', type=int, default=16)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--batch', type=int, default=0)
    ap.add_argument('--count', type=int, default=32)
    ap.add_argument('--input-mode', choices=['sequential', 'affine'], default='sequential')
    ap.add_argument('--check-csv', help='full path to GPU failures.csv')
    args = ap.parse_args()
    assert 3 <= args.pnum <= 65534
    assert (1 << args.ell_i) >= args.pnum and 0 <= args.ell_i <= 61
    batch = batch_encryptor(args.seed, args.batch)
    expected = set()
    for trial in range(args.count):
        oracle = Oracle(args.mode, args.pnum, args.ell_i, batch, trial, args.input_mode)
        if oracle.trial_failed():
            expected.add(trial)
    print(f'CPU oracle: batch={args.batch}, trials={args.count}, failed={len(expected)}, offsets={sorted(expected)}')
    if args.check_csv:
        gpu = set()
        with open(args.check_csv, newline='') as f:
            for rec in csv.DictReader(f):
                if int(rec['batch_id']) == args.batch and int(rec['offset']) < args.count:
                    gpu.add(int(rec['offset']))
        assert gpu == expected, f'FAIL: GPU-only={sorted(gpu - expected)}, CPU-only={sorted(expected - gpu)}'
        print('PASS: GPU/CPU failure offsets agree')


if __name__ == '__main__':
    main()

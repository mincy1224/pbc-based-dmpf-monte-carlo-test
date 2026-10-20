"""Check that smoke-test validation cannot accept missing or mismatched results."""
import csv
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid

import smoke_test as smoke


class SmokeValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = (Path(__file__).resolve().parent / 'build_host' /
                          'smoke_validation_tests' / uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.args = SimpleNamespace(pnum=16, ell_i=8, seed=123, input_mode='sequential')
        buckets, width, domain, _, kicks = smoke.params('drrt18', 16, 8)
        self.summary = dict(mode='drrt18', input_mode='sequential', seed_decimal='123',
                            ET=15, BT=8, PNUM_T=16, ELL_I=8, planned=smoke.TOTAL,
                            BUCKETS_NUM=buckets, BUCKET_SIZE=width, PRP_DOMAIN=domain,
                            MAX_KICK_TIMES=kicks, seed_scheme='aes-batch-v1',
                            status='complete', completed=smoke.TOTAL, failed=2,
                            completed_batches=128, pending_batches=0)
        self.save()

    def save(self, rows=((0, 3), (127, 255))):
        smoke.write_json(self.directory / 'summary.json', self.summary)
        with (self.directory / 'failures.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['batch_id', 'offset'])
            writer.writerows(rows)

    def test_valid_sparse_results_and_full_comparison(self):
        _, failed = smoke.read_results(self.directory, self.args, 'drrt18')
        self.assertEqual(failed, {3, 32767})
        self.assertEqual(smoke.compare(self.directory, failed, dict(gpu=failed)), 0)
        with (self.directory / 'comparison.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), smoke.TOTAL)
        self.assertTrue(all(row['match'] == '1' for row in rows))

    def test_one_wrong_outcome_is_recorded(self):
        self.assertEqual(smoke.compare(self.directory, {3, 32767}, dict(gpu={3})), 1)
        with (self.directory / 'mismatches.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['trial_id'], '32767')
        self.assertEqual((rows[0]['cpu_failed'], rows[0]['gpu_failed'], rows[0]['match']),
                         ('1', '0', '0'))

    def test_bad_sparse_exports_are_rejected(self):
        for rows in (((0, 3), (0, 3)), ((0, 256), (127, 255)), ((0, 3),)):
            with self.subTest(rows=rows):
                self.save(rows)
                with self.assertRaises(RuntimeError):
                    smoke.read_results(self.directory, self.args, 'drrt18')

    def test_incomplete_or_incompatible_run_is_rejected(self):
        for field, value in (('completed', 32767), ('pending_batches', 1),
                             ('status', 'paused'), ('seed_decimal', '124'),
                             ('BUCKETS_NUM', 24)):
            with self.subTest(field=field):
                original = self.summary[field]
                self.summary[field] = value
                self.save()
                with self.assertRaises(RuntimeError):
                    smoke.read_results(self.directory, self.args, 'drrt18')
                self.summary[field] = original


if __name__ == '__main__':
    unittest.main()

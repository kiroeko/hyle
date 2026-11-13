import io
import unittest

from engine.calibration import generated_ids, read_record, write_record
from engine.text.contracts import TokenEvent


class Request:
    def __init__(self, events): self.events = iter(events); self.acks = 0; self.closed = False
    def read(self, timeout=None): return next(self.events)
    def acknowledge(self): self.acks += 1
    def close(self): self.closed = True


class Worker:
    def __init__(self, events): self.request = Request(events)
    def submit(self, prompt, maximum): self.prompt = prompt; return self.request


class CalibrationTest(unittest.TestCase):
    def test_public_generation_acknowledges_ids_and_requires_real_terminal(self):
        worker = Worker([TokenEvent('token', 2), TokenEvent('token', 3), TokenEvent('completed')])
        self.assertEqual(generated_ids(worker, [1], 4, 8), [2, 3])
        self.assertEqual(worker.request.acks, 2)
        self.assertTrue(worker.request.closed)
        for event in (TokenEvent('cancelled'), TokenEvent('error', message='native error'),
                      TokenEvent('turn_end', turn=0), TokenEvent('completed')):
            worker = Worker([event])
            with self.assertRaises(ValueError): generated_ids(worker, [1], 4, 8)
            self.assertTrue(worker.request.closed)

    def test_bad_and_excess_output_do_not_publish_calibration(self):
        for events in ([TokenEvent('token', 8)], [TokenEvent('token', 1), TokenEvent('token', 2)]):
            worker = Worker(events)
            with self.assertRaises(ValueError): generated_ids(worker, [1], 1, 8)
            self.assertTrue(worker.request.closed)

    def test_portable_source_bound_record_and_wrong_source_rejection(self):
        output = io.StringIO(); write_record(output, 'a' * 64, [2, 2, 3])
        self.assertEqual(read_record(output.getvalue(), 'a' * 64, 8), [2, 2, 3])
        for source, vocabulary in (('b' * 64, 8), ('a' * 64, 3)):
            with self.assertRaises(ValueError): read_record(output.getvalue(), source, vocabulary)


if __name__ == '__main__': unittest.main()

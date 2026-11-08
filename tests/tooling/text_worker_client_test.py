import subprocess
import sys
import unittest
from unittest import mock

from engine.text.worker_client import WorkerClient, WorkerError


class WorkerClientLifecycleTest(unittest.TestCase):
    def test_finite_turns_share_one_ordered_request(self):
        code = "\n".join((
            "import sys",
            "print('READY\\t2\\t256\\t128\\t3', flush=True)",
            "line = sys.stdin.readline().rstrip('\\r\\n')",
            "assert line == 'G\\t1\\t1\\t10\\t1\\t\\t1\\t20,21\\t1\\t', repr(line)",
            "print('A\\t1', flush=True)",
            "for token in (11, 12, 13, 14):",
            " print(f'T\\t1\\t{token}', flush=True)",
            " assert sys.stdin.readline().rstrip('\\r\\n') == 'K\\t1\\t1'",
            "print('F\\t1\\tcompleted', flush=True)",
            "assert sys.stdin.readline().rstrip('\\r\\n') == 'Q'",
            "print('BYE', flush=True)",
        ))
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        try:
            with self.assertRaises(ValueError):
                worker.submit_turns([])
            with self.assertRaises(ValueError):
                worker.submit_turns([([], 1)])
            request = worker.submit_turns([([10], 1), ([], 1), ([20, 21], 1), ([], 1)])
            try:
                for token in (11, 12, 13, 14):
                    event = request.read(timeout=3)
                    self.assertEqual((event.kind, event.token), ('token', token))
                    request.acknowledge()
                self.assertEqual(request.read(timeout=3).kind, 'completed')
            finally:
                request.close()
        finally:
            worker.close(timeout=3)

    def test_invalid_handshake_terminates_its_worker(self):
        started = []
        popen = subprocess.Popen
        def record(*args, **kwargs):
            process = popen(*args, **kwargs)
            started.append(process)
            return process
        code = "import time; print('invalid', flush=True); time.sleep(30)"
        with mock.patch('engine.text.worker_client.subprocess.Popen', side_effect=record):
            with self.assertRaises(WorkerError):
                WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].poll())
        if started[0].stdin:
            started[0].stdin.close()
        if started[0].stdout:
            started[0].stdout.close()

    def test_startup_timeout_reaps_its_worker(self):
        started = []
        popen = subprocess.Popen
        def record(*args, **kwargs):
            process = popen(*args, **kwargs)
            started.append(process)
            return process
        with mock.patch('engine.text.worker_client.subprocess.Popen', side_effect=record):
            with self.assertRaises(WorkerError) as error:
                WorkerClient([sys.executable, '-B', '-c', 'import time; time.sleep(30)'], startup_timeout=.1)
        self.assertEqual(error.exception.kind, 'startup')
        self.assertIsNotNone(started[0].poll())

    def test_invalid_live_frame_fails_request_and_stops_worker(self):
        code = ("import sys,time; print('READY\\t2\\t256\\t128\\t3', flush=True); "
                "sys.stdin.readline(); print('invalid', flush=True); time.sleep(30)")
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        request = worker.submit([1], 1)
        try:
            event = request.read(timeout=5)
            self.assertEqual(event.kind, 'error')
            self.assertFalse(worker.healthy)
        finally:
            request.close()
            with self.assertRaises(WorkerError):
                worker.close(timeout=3)
        self.assertIsNotNone(worker._process.poll())

    def test_unexpected_output_eof_stops_worker(self):
        code = ("import os,sys,time; print('READY\\t2\\t256\\t128\\t3', flush=True); "
                "sys.stdin.readline(); fd=sys.stdout.fileno(); sys.stdout.close(); os.close(fd); time.sleep(30)")
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        request = worker.submit([1], 1)
        try:
            event = request.read(timeout=5)
            self.assertEqual(event.kind, 'error')
            self.assertFalse(worker.healthy)
        finally:
            request.close()
            with self.assertRaises(WorkerError):
                worker.close(timeout=1)
        self.assertIsNotNone(worker._process.poll())

    def test_shutdown_timeout_stops_worker_and_reports_failure(self):
        code = ("import sys,time; print('READY\\t2\\t256\\t128\\t3', flush=True); "
                "sys.stdin.readline(); time.sleep(30)")
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        with self.assertRaises(WorkerError) as error:
            worker.close(timeout=.1)
        self.assertEqual(error.exception.kind, 'shutdown')
        self.assertIsNotNone(worker._process.poll())

    def test_nonzero_exit_after_quit_is_not_success(self):
        code = ("import sys; print('READY\\t2\\t256\\t128\\t3', flush=True); "
                "sys.stdin.readline(); sys.exit(7)")
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        with self.assertRaises(WorkerError) as error:
            worker.close(timeout=3)
        self.assertIn('status 7', str(error.exception))

    def test_clean_exit_without_shutdown_acknowledgement_is_not_success(self):
        code = ("import sys; print('READY\\t2\\t256\\t128\\t3', flush=True); "
                "sys.stdin.readline()")
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        with self.assertRaises(WorkerError) as error:
            worker.close(timeout=3)
        self.assertIn('without shutdown acknowledgement', str(error.exception))

    def test_clean_acknowledged_shutdown(self):
        code = ("import sys; print('READY\\t2\\t256\\t128\\t3', flush=True); "
                "sys.stdin.readline(); print('BYE', flush=True)")
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        worker.close(timeout=3)
        self.assertIsNotNone(worker._process.poll())


if __name__ == '__main__':
    unittest.main()

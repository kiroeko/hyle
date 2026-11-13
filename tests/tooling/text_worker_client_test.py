import subprocess
import base64
import sys
import unittest
from unittest import mock

from engine.text.worker_client import WorkerClient, WorkerError


class WorkerClientLifecycleTest(unittest.TestCase):
    def test_prepared_metadata_reports_actual_profiles_and_escaped_names(self):
        name = 'pictures\t图\nprofile'
        encode = lambda value: base64.b64encode(value.encode('utf-8')).decode('ascii')
        frames = ['READY\t4\t256\t128\t3\t' + 'a' * 64 + '\t2',
            'PROFILE\t0\t' + encode('default') + '\t' + encode('ordinary') + '\t0\t' + encode('text') + '\t1',
            'PROFILE\t1\t' + encode(name) + '\t' + encode('proposal') + '\t2\t' + encode('text') + ',' + encode('image') + '\t0']
        code = '\n'.join(('import sys', *(f'print({frame!r}, flush=True)' for frame in frames),
            "assert sys.stdin.readline().rstrip() == 'Q'", "print('BYE', flush=True)"))
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        try:
            self.assertEqual(worker.execution_profiles[0].algorithm, 'ordinary')
            self.assertEqual(worker.execution_profiles[1].name, name)
            self.assertEqual(worker.execution_profiles[1].max_draft_tokens, 2)
            self.assertEqual(worker.execution_profiles[1].input_modalities, ('text', 'image'))
            self.assertEqual(worker.execution_profiles[0].co_resident_with, (1,))
        finally:
            worker.close(timeout=3)

    def test_incomplete_or_invalid_prepared_metadata_never_becomes_ready(self):
        profile = 'PROFILE\t0\tZGVmYXVsdA==\tb3JkaW5hcnk=\t0\tdGV4dA==\t-'
        cases = [
            ['READY\t4\t256\t128\t3\t-\t0'],
            ['READY\t4\t256\t128\t3\t-\t2', profile],
            ['READY\t4\t256\t128\t3\t-\t2', profile, profile.replace('PROFILE\t0', 'PROFILE\t1')],
            ['READY\t4\t256\t128\t3\t-\t1', profile.replace('PROFILE\t0', 'PROFILE\t1')],
            ['READY\t4\t256\t128\t3\t-\t1', profile.replace('\t0\tdGV4dA==', '\t-1\tdGV4dA==')],
            ['READY\t4\t256\t128\t3\t-\t1', profile.replace('dGV4dA==', 'dGV4dA==,dGV4dA==')],
            ['READY\t4\t256\t128\t3\t-\t1', profile.replace('ZGVmYXVsdA==', '@@@@')],
            ['READY\t4\t256\t128\t3\t-\t1', profile[:-1] + '1'],
        ]
        for frames in cases:
            with self.subTest(frames=frames):
                code = '\n'.join(('import sys', *(f'print({frame!r}, flush=True)' for frame in frames)))
                with self.assertRaises(WorkerError) as error:
                    WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
                self.assertEqual(error.exception.kind, 'startup')

    def test_current_ready_reports_original_source_and_unknown_is_explicit(self):
        for identity in ('a' * 64, '-'):
            code = '\n'.join(('import sys',
                f"print('READY\\t3\\t256\\t128\\t3\\t{identity}', flush=True)",
                "assert sys.stdin.readline().rstrip() == 'Q'", "print('BYE', flush=True)"))
            worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
            try:
                self.assertEqual(worker.source_identity, None if identity == '-' else identity)
            finally:
                worker.close(timeout=3)

    def test_pipe_close_failure_is_reported_as_shutdown_error(self):
        code = "\n".join((
            "import sys",
            "print('READY\\t2\\t256\\t128\\t3', flush=True)",
            "assert sys.stdin.readline().rstrip() == 'Q'",
            "print('BYE', flush=True)",
        ))
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        stream = worker._process.stdin

        class BrokenClose:
            def write(self, value): return stream.write(value)
            def flush(self): return stream.flush()
            def close(self): raise OSError('pipe close failed')

        worker._process.stdin = BrokenClose()
        try:
            with self.assertRaises(WorkerError) as error:
                worker.close(timeout=3)
            self.assertEqual(error.exception.kind, 'shutdown')
            self.assertIn('pipe', str(error.exception))
        finally:
            worker._process.stdin = stream
            stream.close()

    def test_worker_failure_is_observable_after_full_event_queue_drains(self):
        code = "\n".join((
            "import sys,time",
            "print('READY\\t2\\t256\\t128\\t3', flush=True)",
            "sys.stdin.readline()",
            "print('T\\t1\\t11', flush=True)",
            "print('B\\t1\\t0', flush=True)",
            "print('invalid', flush=True)",
            "time.sleep(30)",
        ))
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        request = worker.open_session([10], 1)
        try:
            worker._reader.join(timeout=3)
            self.assertFalse(worker._reader.is_alive(), 'worker output failure was not observed')
            # An acknowledgement already accepted by the pipe can race with
            # worker termination. Isolate event delivery from that timing.
            with mock.patch.object(worker, '_send'):
                self.assertEqual(request.read(timeout=1).token, 11); request.acknowledge()
                self.assertEqual(request.read(timeout=1).kind, 'turn_end'); request.acknowledge()
                event = request.read(timeout=1)
            self.assertEqual(event.kind, 'error')
            self.assertTrue(event.message)
        finally:
            request.close()
            with self.assertRaises(WorkerError):
                worker.close(timeout=3)

    def test_invalid_worker_output_fails_all_pending_requests(self):
        for frame in ('T\t1\t-1', 'T\t1\t256', 'T\t0\t1', 'B\t-1\t0',
                      'F\t0\tcompleted', 'E\t-1\tinvalid\tbad request'):
            with self.subTest(frame=frame):
                code = "\n".join((
                    "import sys,time",
                    "read = lambda: sys.stdin.readline().rstrip('\\r\\n')",
                    "print('READY\\t2\\t256\\t128\\t3', flush=True)",
                    "assert read() == 'G\\t1\\t1\\t10'",
                    "assert read() == 'G\\t2\\t1\\t11'",
                    f"print({frame!r}, flush=True)",
                    "time.sleep(10)",
                ))
                worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
                first = worker.submit([10], 1)
                second = worker.submit([11], 1)
                try:
                    for request in (first, second):
                        event = request.read(timeout=3)
                        self.assertEqual(event.kind, 'error')
                        self.assertTrue(event.message)
                    self.assertFalse(worker.healthy)
                finally:
                    first.close(); second.close()
                    with self.assertRaises(WorkerError):
                        worker.close(timeout=3)

    def test_request_error_kind_remains_separate_from_its_message(self):
        code = "\n".join((
            "import sys",
            "read = lambda: sys.stdin.readline().rstrip('\\r\\n')",
            "print('READY\\t2\\t256\\t128\\t3', flush=True)",
            "assert read() == 'G\\t1\\t1\\t10'",
            "print('A\\t1', flush=True); print('E\\t1\\tinvalid\\tbad request', flush=True)",
            "assert read() == 'Q'; print('BYE', flush=True)",
        ))
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        try:
            request = worker.submit([10], 1)
            try:
                event = request.read(timeout=3)
                self.assertEqual((event.kind, event.error_kind, event.message), ('error', 'invalid', 'bad request'))
            finally:
                request.close()
        finally:
            worker.close(timeout=3)

    def test_extra_stop_tokens_are_sent_per_request_in_each_mode(self):
        for kind in ('g', 's', 'o'):
            with self.subTest(kind=kind):
                code = "\n".join((
                    "import sys",
                    "read = lambda: sys.stdin.readline().rstrip('\\r\\n')",
                    "print('READY\\t2\\t256\\t128\\t3', flush=True)",
                    f"assert read() == '{kind}\\t1\\t4,8\\t1\\t10'",
                    "print('A\\t1', flush=True); print('T\\t1\\t4', flush=True)",
                    "assert read() == 'K\\t1\\t1'; print('F\\t1\\tcompleted', flush=True)",
                    "assert read() == 'Q'; print('BYE', flush=True)",
                ))
                worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
                try:
                    with self.assertRaises(ValueError):
                        worker.submit([10], 1, stop_ids=[256])
                    if kind == 'g':
                        request = worker.submit([10], 1, stop_ids=[8, 3, 4, 8])
                    elif kind == 's':
                        request = worker.submit_turns([([10], 1)], stop_ids=[8, 3, 4, 8])
                    else:
                        request = worker.open_session([10], 1, stop_ids=[8, 3, 4, 8])
                    try:
                        self.assertEqual(request.read(timeout=3).token, 4)
                        request.acknowledge()
                        self.assertEqual(request.read(timeout=3).kind, 'completed')
                    finally:
                        request.close()
                finally:
                    worker.close(timeout=3)

    def test_request_configuration_is_transmitted_in_each_mode(self):
        for kind, profile in ((kind, profile) for kind in ('G', 'S', 'O') for profile in (None, 2)):
            with self.subTest(kind=kind, execution_profile=profile):
                fields = ['R' if profile is None else 'P', '1', kind]
                if profile is not None:
                    fields.append(str(profile))
                expected = '\t'.join([*fields, '2', '0', '3', '4,8', '1', '10'])
                code = "\n".join((
                    "import sys",
                    "read = lambda: sys.stdin.readline().rstrip('\\r\\n')",
                    "print('READY\\t2\\t256\\t128\\t3', flush=True)",
                    f"assert read() == {expected!r}",
                    "print('A\\t1', flush=True); print('T\\t1\\t4', flush=True)",
                    "assert read() == 'K\\t1\\t1'; print('F\\t1\\tcompleted', flush=True)",
                    "assert read() == 'Q'; print('BYE', flush=True)",
                ))
                worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
                try:
                    for kwargs in ({'seed': -1}, {'seed': True}, {'sampling_index': 2**64},
                                   {'max_draft_tokens': -1}, {'execution_profile': -1}, {'execution_profile': True}):
                        with self.assertRaises(ValueError):
                            worker.submit([10], 1, **kwargs)
                    options = dict(stop_ids=[8, 3, 4], sampling_index=2, seed=0, max_draft_tokens=3,
                                   execution_profile=profile)
                    if kind == 'G':
                        request = worker.submit([10], 1, **options)
                    elif kind == 'S':
                        request = worker.submit_turns([([10], 1)], **options)
                    else:
                        request = worker.open_session([10], 1, **options)
                    try:
                        self.assertEqual(request.read(timeout=3).token, 4)
                        request.acknowledge()
                        self.assertEqual(request.read(timeout=3).kind, 'completed')
                    finally:
                        request.close()
                finally:
                    worker.close(timeout=3)

    def test_timed_out_input_cannot_be_retried_as_if_rejected(self):
        code = "\n".join((
            "import sys,time",
            "read = lambda: sys.stdin.readline().rstrip('\\r\\n')",
            "print('READY\\t2\\t256\\t128\\t3', flush=True)",
            "assert read() == 'O\\t1\\t1\\t10'; print('A\\t1', flush=True)",
            "assert read() == 'I\\t1\\t1\\t1\\t20'; time.sleep(.2)",
            "print('I\\t1\\t1\\taccepted', flush=True)",
            "assert read() == 'C\\t1'; print('C\\t1\\taccepted', flush=True)",
            "print('F\\t1\\tcancelled', flush=True)",
            "assert read() == 'Q'; print('BYE', flush=True)",
        ))
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        try:
            session = worker.open_session([10], 1)
            try:
                with self.assertRaises(WorkerError) as timeout:
                    session.append([20], 1, timeout=.02)
                self.assertEqual(timeout.exception.kind, 'timeout')
                with self.assertRaises(WorkerError) as uncertain:
                    session.append([20], 1, timeout=3)
                self.assertEqual(uncertain.exception.kind, 'uncertain')
            finally:
                session.close()
        finally:
            worker.close(timeout=3)

    def test_live_session_appends_with_correlated_input_acknowledgements(self):
        code = "\n".join((
            "import sys",
            "read = lambda: sys.stdin.readline().rstrip('\\r\\n')",
            "print('READY\\t2\\t256\\t128\\t3', flush=True)",
            "assert read() == 'O\\t1\\t1\\t10'",
            "print('A\\t1', flush=True); print('T\\t1\\t11', flush=True)",
            "assert read() == 'K\\t1\\t1'; print('B\\t1\\t0', flush=True)",
            "assert read() == 'I\\t1\\t1\\t1\\t256'; print('I\\t1\\t1\\tinvalid', flush=True)",
            "assert read() == 'I\\t1\\t2\\t1\\t20'; print('I\\t1\\t2\\tfailed', flush=True)",
            "assert read() == 'I\\t1\\t3\\t1\\t20'; print('I\\t1\\t3\\taccepted', flush=True)",
            "assert read() == 'Z\\t1\\t4'; print('Z\\t1\\t4\\taccepted', flush=True)",
            "assert read() == 'K\\t1\\t1'; print('T\\t1\\t21', flush=True)",
            "assert read() == 'K\\t1\\t1'; print('B\\t1\\t1', flush=True)",
            "assert read() == 'K\\t1\\t1'; print('F\\t1\\tcompleted', flush=True)",
            "assert read() == 'Q'; print('BYE', flush=True)",
        ))
        worker = WorkerClient([sys.executable, '-B', '-c', code], startup_timeout=3)
        try:
            session = worker.open_session([10], 1)
            try:
                self.assertEqual(session.read(timeout=3).token, 11); session.acknowledge()
                self.assertEqual(session.read(timeout=3).turn, 0)
                with self.assertRaises(WorkerError) as error:
                    session.append([256], 1, timeout=3)
                self.assertEqual(error.exception.kind, 'invalid')
                with self.assertRaises(WorkerError) as failed:
                    session.append([20], 1, timeout=3)
                self.assertEqual(failed.exception.kind, 'failed')
                session.append([20], 1, timeout=3); session.finish_input(timeout=3)
                with self.assertRaises(WorkerError):
                    session.append([30], 1, timeout=3)
                session.acknowledge()
                self.assertEqual(session.read(timeout=3).token, 21); session.acknowledge()
                self.assertEqual(session.read(timeout=3).turn, 1); session.acknowledge()
                self.assertEqual(session.read(timeout=3).kind, 'completed')
            finally:
                session.close()
        finally:
            worker.close(timeout=3)

    def test_finite_turns_share_one_ordered_request(self):
        code = "\n".join((
            "import sys",
            "print('READY\\t2\\t256\\t128\\t3', flush=True)",
            "line = sys.stdin.readline().rstrip('\\r\\n')",
            "assert line == 'S\\t1\\t1\\t10\\t1\\t\\t1\\t20,21\\t1\\t', repr(line)",
            "print('A\\t1', flush=True)",
            "for kind, value in (('T', 11), ('B', 0), ('T', 12), ('B', 1), ('T', 13), ('B', 2), ('T', 14), ('B', 3)):",
            " print(f'{kind}\\t1\\t{value}', flush=True)",
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
                    boundary = request.read(timeout=3)
                    self.assertEqual((boundary.kind, boundary.turn), ('turn_end', token - 11))
                    with self.assertRaises(RuntimeError):
                        request.read(timeout=3)
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
